#ifndef CVFEM_HEX8_BEST_PACKED_HPP
#define CVFEM_HEX8_BEST_PACKED_HPP

// Packed layout: elements are grouped into packs, each pack accumulates into a
// thread-private buffer indexed by pack-local node ids, and the buffer is folded
// back into the global structure afterwards. Ghost rows -- nodes a pack touches
// but does not own -- are reduced in a second pass.
//
// The pack-local indexing lets the residual and Jacobian-action run 16-wide SIMD
// over elements. For assembly the local matrix is large enough that the round
// trip through it costs more than it saves; see cvfem_hex8_best_colored.hpp
// and cvfem_hex8_best_store.hpp.

#include "kernels/microkernels/hex8/affine/cvfem_hex8_ns_upwind_affine.hpp"
#include "kernels/microkernels/hex8/isoparametric/cvfem_hex8_ns_upwind_isoparam.hpp"
#include "kernels/cvfem_scatter.hpp"
#include "kernels/cvfem_phases.hpp"
#include "kernels/cvfem_range.hpp"
#include "kernels/packed/cvfem_hex8_pack_staging.hpp"
#include "kernels/cvfem_hex8_flags.hpp"









// ONE ELEMENT, OUT OF THE PACK. The assembly sweeps work scalar per element rather than
// lane-blocked, so both geometries gather the same four nodal fields, pick the same slot array
// and build the same Hex8RhieChow before they diverge -- and they diverge only in where the
// geometry comes from and which kernel they then call.
//
// It is a struct rather than six out-parameters because Hex8RhieChow POINTS AT the coordinate
// arrays: they have to outlive the kernel call, so they live here and `rc` refers to this
// object's own storage. That makes the struct non-copyable in practice -- a copy's `rc` would
// point at the original's arrays -- so it is built in place, by reference, and never returned
// by value.
template <typename scalar_t>
struct Hex8PackElementT {
    scalar_t     ux[8], uy[8], uz[8], p[8];
    scalar_t     rc_x[8], rc_y[8], rc_z[8], rc_pgx[8], rc_pgy[8], rc_pgz[8];
    Hex8RhieChowT<scalar_t> rc;
    // The pressure the Rhie-Chow term differences, or null when the term is off -- which is
    // what makes the kernel's own branch on it fold away.
    const scalar_t *rc_p;
};

using Hex8PackElement = Hex8PackElementT<scalar_t>;

template <typename scalar_t, typename pack_idx_t>
static SFEM_INLINE void cvfem_hex8_stage_pack_element(pack_idx_t **const SFEM_RESTRICT pack_elems,
                                                      const scalar_t *const SFEM_RESTRICT pack_u,
                                                      const Hex8PackCoordsT<scalar_t>    &pk,
                                                      const ptrdiff_t                     e,
                                                      const int                           with_rc,
                                                      const Hex8RcConfig                 &rc_cfg,
                                                      Hex8PackElementT<scalar_t>         &el) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const scalar_t *const SFEM_RESTRICT u =
                pack_u + (ptrdiff_t)pack_elems[a][e] * CVFEM_HEX8_N_FIELDS;
        el.ux[a] = u[0];
        el.uy[a] = u[1];
        el.uz[a] = u[2];
        el.p[a]  = u[3];
    }
    el.rc   = Hex8RhieChowT<scalar_t>{};
    el.rc_p = nullptr;
    if (with_rc) {
        gather_hex8_coords_from_pack(pack_elems, pk.x, pk.y, pk.z, e, el.rc_x, el.rc_y, el.rc_z);
        gather_hex8_coords_from_pack(pack_elems, pk.pgx, pk.pgy, pk.pgz, e, el.rc_pgx, el.rc_pgy, el.rc_pgz);
        el.rc   = Hex8RhieChowT<scalar_t>{el.rc_x, el.rc_y,  el.rc_z,  el.rc_pgx, el.rc_pgy, el.rc_pgz,
                               rc_cfg.scale, nullptr, nullptr, nullptr, el.ux, el.uy, el.uz, rc_cfg.tau};
        el.rc_p = el.p;
    }
}












#endif  // CVFEM_HEX8_BEST_PACKED_HPP
