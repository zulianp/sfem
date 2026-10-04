#ifndef CVFEM_HEX8_NS_PACKED_LAUNCH_HPP
#define CVFEM_HEX8_NS_PACKED_LAUNCH_HPP

// THE SOLVER'S PACKED LAUNCHERS, AND ITS TWO MESH-TAKING WRAPPERS.
//
// cvfem_hex8_precompute_affine_geometry and cvfem_hex8_load_adj are one-line wrappers that exist
// so this family can call the shared versions with its own mesh type; they read a mesh, so they
// belong here. The rest are the launchers: the residual, the Jacobian action, and the three
// ghost reductions, each resolving the arrays and owning the parallel region whose range-taking
// kernel stays behind.
//
// This was the last file in src/kernels/ reaching outside the directory -- for
// smesh_packed_mesh.hpp and the two pack staging headers, all of which these functions need and
// the kernels do not.
#include "frontend/staging/cvfem_hex8_pack_common.hpp"
#include "frontend/staging/cvfem_hex8_pack_helpers.hpp"
#include "kernels/packed/cvfem_hex8_ns_packed.hpp"
#include "smesh_packed_mesh.hpp"

static void cvfem_hex8_precompute_affine_geometry(MeshData &d) {
    precompute_affine_geometry(d);
}

static SFEM_INLINE void cvfem_hex8_load_adj(const MeshData &d, const ptrdiff_t e, scalar_t adj[9], scalar_t *det) {
    load_hex8_adj(d.adj_ptr, d.det_ptr, e, adj, det);
}

static SFEM_INLINE void cvfem_hex8_ghost_reduce_soa(PackedData &p, scalar_t *const fields[CVFEM_HEX8_N_FIELDS]) {
#pragma omp parallel
    cvfem_hex8_ghost_reduce_soa_range(cvfem_range_split(0, p.n_ghost_reduce_rows, 1,
                                   cvfem_thread_index(), cvfem_n_threads()),
            p.ghost_reduce_dest, p.ghost_reduce_ptr, p.ghost_reduce_idx,
            p.n_ghost_entries, p.ghost_buf.data(), fields);
}

static SFEM_INLINE void cvfem_hex8_ghost_reduce_wide(PackedData                         &p,
        const scalar_t *const SFEM_RESTRICT buf,
        const int                           width,
        scalar_t *const SFEM_RESTRICT       dst) {
#pragma omp parallel
    cvfem_hex8_ghost_reduce_wide_range(cvfem_range_split(0, p.n_ghost_reduce_rows, 1,
                                   cvfem_thread_index(), cvfem_n_threads()),
            p.ghost_reduce_dest, p.ghost_reduce_ptr, p.ghost_reduce_idx,
            p.n_ghost_entries, buf, width, dst);
}

static SFEM_INLINE void cvfem_hex8_ghost_reduce_interleaved(PackedData &p, scalar_t *const SFEM_RESTRICT jv) {
#pragma omp parallel
    cvfem_hex8_ghost_reduce_interleaved_range(cvfem_range_split(0, p.n_ghost_reduce_rows, 1,
                                   cvfem_thread_index(), cvfem_n_threads()),
            p.ghost_reduce_dest, p.ghost_reduce_ptr, p.ghost_reduce_idx,
            p.n_ghost_entries, p.ghost_buf.data(), jv);
}

static SFEM_NOINLINE void cvfem_hex8_apply_residual_packed(MeshData &d, PackedData &p, const scalar_t rho, const scalar_t mu) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_residual_packed");
    const size_t scratch_n = packed_scratch_n(p.max_actual_nodes_per_pack);
    const size_t rc_n      = packed_rc_n(p.max_actual_nodes_per_pack);
    const int    with_rc   = d.rhie_chow_scale != scalar_t(0);


#pragma omp parallel
    cvfem_hex8_apply_residual_packed_range(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d.adj_ptr, d.det_ptr, d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.rc.data(), d.rhie_chow_scale, d.rx.data(), d.ry.data(), d.rz.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), p.elems, p.ghost_buf.data(), p.ghost_idx, p.ghost_ptr, p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.n_ghost_entries, p.owned_nodes_ptr, rho, mu, scratch_n, rc_n, with_rc);


    scalar_t *const fields[CVFEM_HEX8_N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};
    cvfem_hex8_ghost_reduce_soa(p, fields);
}

static SFEM_NOINLINE void cvfem_hex8_apply_jacobian_action_packed(MeshData                    &d,
                                                                  PackedData                  &p,
                                                                  const scalar_t               rho,
                                                                  const scalar_t               mu,
                                                                  const scalar_t *const SFEM_RESTRICT dir,
                                                                  scalar_t *const SFEM_RESTRICT       jv) {
    {
        // Hoisted out of the face loops -- see Hex8RhieChowPack::coeff. Guarded by its own
        // cache key, so this is a handful of comparisons on every call but the first after
        // rho, mu or the scale move; the Reynolds continuation moves mu between stages,
        // which is why it lives here and not in initialize().
        SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::build_rc_coeff");
        cvfem_hex8_build_rc_coeff(d, rho, mu);
    }
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_jacobian_action_packed");
    const size_t scratch_n = packed_scratch_n(p.max_actual_nodes_per_pack);
    const size_t rc_n      = packed_rc_n(p.max_actual_nodes_per_pack);
    const size_t qg_n      = packed_qg_n(p.max_actual_nodes_per_pack);
    const int    with_rc   = d.rhie_chow_scale != scalar_t(0);
    // The Rhie-Chow term differentiates through the nodal pressure-gradient reconstruction.
    // apply_jacobian_action_accumulate reconstructs the direction's gradient into d.qg
    // before calling this, or clears it, so a non-empty d.qgx is exactly the signal that the
    // exact form is wanted. Without this the packed Jacobian is the frozen-pg one while the
    // residual is not, and Newton is capped at a linear rate.
    const bool   with_qg   = with_rc && !d.qgx.empty();
    // The exact higher-order action, signalled the same way: a non-empty d.vgrad means
    // apply_jacobian_action_accumulate reconstructed the direction's velocity gradient for it.
    const bool   with_ho   = d.conv_ho != 0 && !d.ugrad.empty() && !d.vgrad.empty();


#pragma omp parallel
    cvfem_hex8_apply_jacobian_action_packed_range(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d.adj_ptr, d.conv_limiter, d.conv_venkat_c, d.det_ptr, d.elems, d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc_coeff.data(), d.rc_w.data(), d.rhie_chow_scale, d.ugrad.data(), d.upwind_eps, d.ux.data(), d.uy.data(), d.uz.data(), d.vgrad.data(), p.elems, p.ghost_buf.data(), p.ghost_idx, p.ghost_ptr, p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.n_ghost_entries, p.owned_nodes_ptr, rho, mu, dir, jv, scratch_n, rc_n, qg_n, with_rc, with_qg, with_ho,
            cvfem_hex8_rc_config_for(d));


    cvfem_hex8_ghost_reduce_interleaved(p, jv);
}

#endif  // CVFEM_HEX8_NS_PACKED_LAUNCH_HPP
