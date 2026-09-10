#ifndef CVFEM_HEX8_PACK_HELPERS_HPP
#define CVFEM_HEX8_PACK_HELPERS_HPP

// Geometry and pack-staging helpers shared by the two packed implementations.
//
// The benchmark (cvfem_hex8_layout_*.hpp) and the Newton solver
// (cvfem_hex8_ns_packed.hpp) each carry their own MeshData: the solver's is the
// NOT self-contained: include it after the CVFEM kernel headers and after scalar_t /
// MeshData are in scope. It uses CVFEM_HEX8_N_NODES, CVFEM_HEX8_VEC_SIZE,
// Hex8ResidualPack and cvfem_hex8_affine_adj.
//
// benchmark's plus a domain size, a nodal pressure gradient and a Rhie-Chow scale. That
// is a real difference and not worth forcing into one type, so these helpers are
// templated on the mesh type instead and both callers pass their own.
//
// Only the parts that were textually identical live here. The two apply_residual_packed
// implementations are NOT duplicates -- the solver's carries Rhie-Chow and boundary
// terms -- and stay where they are. Measured before this was written: of the solver
// header's 546 lines, 83 (15%) were duplicated, 68 (12%) are Rhie-Chow staging the
// benchmark has no use for, and 370 (68%) genuinely differ.

template <typename MeshT>
static SFEM_INLINE void load_hex8_adj(const MeshT &d, const ptrdiff_t e, scalar_t adj[9], scalar_t *det) {
    for (int c = 0; c < 9; ++c) adj[c] = d.jacobian_adjugate[c][(size_t)e];
    *det = d.jacobian_determinant[(size_t)e];
}

template <typename MeshT>
static void precompute_affine_geometry(MeshT &d) {
    for (int c = 0; c < 9; ++c) d.jacobian_adjugate[c].resize((size_t)d.nelements);
    d.jacobian_determinant.resize((size_t)d.nelements);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], adj[9], det;
        const auto *const px = d.points[0];
        const auto *const py = d.points[1];
        const auto *const pz = d.points[2];
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            x[a]                 = scalar_t(px[g]);
            y[a]                 = scalar_t(py[g]);
            z[a]                 = scalar_t(pz[g]);
        }
        cvfem_hex8_affine_adj(x, y, z, adj, &det);
        for (int c = 0; c < 9; ++c) d.jacobian_adjugate[c][(size_t)e] = adj[c];
        d.jacobian_determinant[(size_t)e] = det;
    }
}

// No mesh argument, so no template parameter to deduce -- a plain function.
static SFEM_INLINE void scatter_hex8_simd_to_pack(pack_idx_t **const SFEM_RESTRICT elems,
                                                  scalar_t *const SFEM_RESTRICT    pack_out,
                                                  const ptrdiff_t                  begin,
                                                  const int                        nlanes,
                                                  const Hex8ResidualPack          &out) {
    for (int lane = 0; lane < nlanes; ++lane) {
        const ptrdiff_t e = begin + lane;
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            scalar_t *const SFEM_RESTRICT dst = pack_out + (ptrdiff_t)elems[a][e] * N_FIELDS;
            dst[0] += out.rx[a][lane];
            dst[1] += out.ry[a][lane];
            dst[2] += out.rz[a][lane];
            dst[3] += out.rc[a][lane];
        }
    }
}

template <typename MeshT>
static SFEM_INLINE void gather_hex8_adj_soa(const MeshT               &d,
                                            const ptrdiff_t               begin,
                                            const int                     nlanes,
                                            scalar_t *const SFEM_RESTRICT cof0,
                                            scalar_t *const SFEM_RESTRICT cof1,
                                            scalar_t *const SFEM_RESTRICT cof2,
                                            scalar_t *const SFEM_RESTRICT cof3,
                                            scalar_t *const SFEM_RESTRICT cof4,
                                            scalar_t *const SFEM_RESTRICT cof5,
                                            scalar_t *const SFEM_RESTRICT cof6,
                                            scalar_t *const SFEM_RESTRICT cof7,
                                            scalar_t *const SFEM_RESTRICT cof8,
                                            scalar_t *const SFEM_RESTRICT det) {
    const size_t n = (size_t)nlanes * sizeof(scalar_t);
    std::memcpy(cof0, d.jacobian_adjugate[0].data() + begin, n);
    std::memcpy(cof1, d.jacobian_adjugate[1].data() + begin, n);
    std::memcpy(cof2, d.jacobian_adjugate[2].data() + begin, n);
    std::memcpy(cof3, d.jacobian_adjugate[3].data() + begin, n);
    std::memcpy(cof4, d.jacobian_adjugate[4].data() + begin, n);
    std::memcpy(cof5, d.jacobian_adjugate[5].data() + begin, n);
    std::memcpy(cof6, d.jacobian_adjugate[6].data() + begin, n);
    std::memcpy(cof7, d.jacobian_adjugate[7].data() + begin, n);
    std::memcpy(cof8, d.jacobian_adjugate[8].data() + begin, n);
    std::memcpy(det, d.jacobian_determinant.data() + begin, n);
    if (nlanes < CVFEM_HEX8_VEC_SIZE) {
        const size_t pad = (size_t)(CVFEM_HEX8_VEC_SIZE - nlanes) * sizeof(scalar_t);
        std::memset(cof0 + nlanes, 0, pad);
        std::memset(cof1 + nlanes, 0, pad);
        std::memset(cof2 + nlanes, 0, pad);
        std::memset(cof3 + nlanes, 0, pad);
        std::memset(cof4 + nlanes, 0, pad);
        std::memset(cof5 + nlanes, 0, pad);
        std::memset(cof6 + nlanes, 0, pad);
        std::memset(cof7 + nlanes, 0, pad);
        std::memset(cof8 + nlanes, 0, pad);
        for (int lane = nlanes; lane < CVFEM_HEX8_VEC_SIZE; ++lane) det[lane] = scalar_t(1);
    }
}

// ---------------------------------------------------------------- Rhie-Chow pack staging
//
// Moved here from cvfem_hex8_ns_packed.hpp so the benchmark can stage the term too. The
// gather needs nothing but raw arrays and was already family-independent; the filler is
// templated on the two container types the way the rest of this header is.
static SFEM_INLINE void cvfem_hex8_gather_rc_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                       const scalar_t *const SFEM_RESTRICT pack_x,
                                                       const scalar_t *const SFEM_RESTRICT pack_y,
                                                       const scalar_t *const SFEM_RESTRICT pack_z,
                                                       const scalar_t *const SFEM_RESTRICT pack_pgx,
                                                       const scalar_t *const SFEM_RESTRICT pack_pgy,
                                                       const scalar_t *const SFEM_RESTRICT pack_pgz,
                                                       const ptrdiff_t                     begin,
                                                       const int                           nlanes,
                                                       Hex8RhieChowPack                   &rc) {
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t loc = elems[a][e];
                rc.x[a][lane]        = pack_x[loc];
                rc.y[a][lane]        = pack_y[loc];
                rc.z[a][lane]        = pack_z[loc];
                rc.pgx[a][lane]      = pack_pgx[loc];
                rc.pgy[a][lane]      = pack_pgy[loc];
                rc.pgz[a][lane]      = pack_pgz[loc];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                rc.x[a][lane] = rc.y[a][lane] = rc.z[a][lane] = scalar_t(0);
                rc.pgx[a][lane] = rc.pgy[a][lane] = rc.pgz[a][lane] = scalar_t(0);
            }
        }
    }
}

// The direction's gradient into the same pack, called straight after the routine above
// when the Jacobian action needs it. Padding lanes are zeroed here too: they multiply real
// geometry and would otherwise contribute whatever the last sweep left behind.
static SFEM_INLINE void cvfem_hex8_gather_qg_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                       const scalar_t *const SFEM_RESTRICT pack_qgx,
                                                       const scalar_t *const SFEM_RESTRICT pack_qgy,
                                                       const scalar_t *const SFEM_RESTRICT pack_qgz,
                                                       const ptrdiff_t                     begin,
                                                       const int                           nlanes,
                                                       Hex8RhieChowPack                   &rc) {
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t loc = elems[a][e];
                rc.qgx[a][lane]      = pack_qgx[loc];
                rc.qgy[a][lane]      = pack_qgy[loc];
                rc.qgz[a][lane]      = pack_qgz[loc];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a)
                rc.qgx[a][lane] = rc.qgy[a][lane] = rc.qgz[a][lane] = scalar_t(0);
        }
    }
}

// ------------------------------------------------------- hoisted Rhie-Chow coefficient
//
// The Rhie-Chow mass-flux coefficient is pure geometry -- it depends on the element's
// sub-control-surface area vectors and edge vectors, on rho and mu, and on the scale, and
// on nothing that changes inside a Krylov solve. Building it here once per element and
// reading it in the face loops is worth 1.83x on the packed Jacobian action; the reason is
// in the comment on Hex8RhieChowPack::coeff, and it is about the compiler's vectoriser
// rather than about the arithmetic.
//
// Affine only, and deliberately so. The isoparametric kernels build their area vectors per
// sub-control surface from a trilinear Jacobian, so a table indexed by element would not
// describe what they evaluate; they call cvfem_hex8_rhie_chow_mdot_coeff directly and keep
// the guard inline, which costs them nothing they were going to get -- those kernels take
// no Rhie-Chow argument on the SIMD path at all.
//
// Stored as twelve arrays of nelements rather than one array of twelve, so the gather below
// is the same strided SoA read as gather_hex8_adj_soa and not a stride-12 walk.
template <typename MeshT>
static void cvfem_hex8_build_rc_coeff(MeshT &d, const scalar_t rho, const scalar_t mu) {
    if (d.rhie_chow_scale == scalar_t(0)) {
        for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) d.rc_coeff[s].clear();
        return;
    }
    // Rebuilt only when something it depends on moves. rho and mu do move -- the Reynolds
    // continuation walks mu down between stages -- so this cannot be built once at setup
    // and forgotten, and it must not be rebuilt on every matvec either.
    if (!d.rc_coeff[0].empty() && d.rc_coeff_rho == rho && d.rc_coeff_mu == mu &&
        d.rc_coeff_scale == d.rhie_chow_scale && (ptrdiff_t)d.rc_coeff[0].size() == d.nelements)
        return;
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) d.rc_coeff[s].resize((size_t)d.nelements);
    d.rc_coeff_rho   = rho;
    d.rc_coeff_mu    = mu;
    d.rc_coeff_scale = d.rhie_chow_scale;

    const auto *const px = d.points[0];
    const auto *const py = d.points[1];
    const auto *const pz = d.points[2];

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t adj[9], det, A[3][3];
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_dir_areas(adj, A);
        for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
            const int i = CVFEM_HEX8_SCS[s].i;
            const int j = CVFEM_HEX8_SCS[s].j;
            const int q = s >> 2;
            const smesh::idx_t gi = d.elems[i][e];
            const smesh::idx_t gj = d.elems[j][e];
            // The guard the face loops no longer carry runs right here, inside
            // cvfem_hex8_rhie_chow_mdot_coeff -- a degenerate sub-control surface still
            // yields exactly zero, and the scalar paths get the same value from the same
            // function.
            d.rc_coeff[s][(size_t)e] = cvfem_hex8_rhie_chow_mdot_coeff(
                    rho, mu, d.rhie_chow_scale,
                    scalar_t(px[gj]) - scalar_t(px[gi]),
                    scalar_t(py[gj]) - scalar_t(py[gi]),
                    scalar_t(pz[gj]) - scalar_t(pz[gi]),
                    A[q][0], A[q][1], A[q][2]);
        }
    }
}

// The SoA gather for the above, straight into the pack the face loops read.
template <typename MeshT>
static SFEM_INLINE void cvfem_hex8_gather_rc_coeff(const MeshT      &d,
                                                   const ptrdiff_t   begin,
                                                   const int         nlanes,
                                                   Hex8RhieChowPack &rc) {
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const scalar_t *const SFEM_RESTRICT src = d.rc_coeff[s].data();
        for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane)
            rc.coeff[s][lane] = lane < nlanes ? src[begin + lane] : scalar_t(0);
    }
}

static SFEM_INLINE void cvfem_hex8_scatter_simd_to_pack(pack_idx_t **const SFEM_RESTRICT elems,
                                                        scalar_t *const SFEM_RESTRICT    pack_out,
                                                        const ptrdiff_t                  begin,
                                                        const int                        nlanes,
                                                        const Hex8ResidualPack          &out) {
    scatter_hex8_simd_to_pack(elems, pack_out, begin, nlanes, out);
}

template <typename PackT, typename MeshT>
static SFEM_INLINE void cvfem_hex8_fill_pack_xyz_pgrad(const PackT                       &p,
                                                       const MeshT                       &d,
                                                       const ptrdiff_t                    pack,
                                                       const ptrdiff_t                    n_contiguous,
                                                       const ptrdiff_t                    n_ghost,
                                                       const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                                       scalar_t *const SFEM_RESTRICT      pack_x,
                                                       scalar_t *const SFEM_RESTRICT      pack_y,
                                                       scalar_t *const SFEM_RESTRICT      pack_z,
                                                       scalar_t *const SFEM_RESTRICT      pack_pgx,
                                                       scalar_t *const SFEM_RESTRICT      pack_pgy,
                                                       scalar_t *const SFEM_RESTRICT      pack_pgz) {
    const auto *const px    = d.points[0];
    const auto *const py    = d.points[1];
    const auto *const pz    = d.points[2];
    const ptrdiff_t   owned = p.owned_nodes_ptr[pack];
    const int         with_pg = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        const ptrdiff_t g = owned + k;
        pack_x[k]         = scalar_t(px[g]);
        pack_y[k]         = scalar_t(py[g]);
        pack_z[k]         = scalar_t(pz[g]);
        pack_pgx[k]       = with_pg ? d.pgx[(size_t)g] : scalar_t(0);
        pack_pgy[k]       = with_pg ? d.pgy[(size_t)g] : scalar_t(0);
        pack_pgz[k]       = with_pg ? d.pgz[(size_t)g] : scalar_t(0);
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        const smesh::idx_t g         = ghosts[k];
        pack_x[n_contiguous + k]     = scalar_t(px[g]);
        pack_y[n_contiguous + k]     = scalar_t(py[g]);
        pack_z[n_contiguous + k]     = scalar_t(pz[g]);
        pack_pgx[n_contiguous + k]   = with_pg ? d.pgx[(size_t)g] : scalar_t(0);
        pack_pgy[n_contiguous + k]   = with_pg ? d.pgy[(size_t)g] : scalar_t(0);
        pack_pgz[n_contiguous + k]   = with_pg ? d.pgz[(size_t)g] : scalar_t(0);
    }
}

// The same staging for the DIRECTION's reconstructed gradient, which only the Jacobian
// action needs. Separate from the routine above rather than another pair of arguments on
// it: the residual and the benchmark call that one and have nothing to put here.
template <typename PackT, typename MeshT>
static SFEM_INLINE void cvfem_hex8_fill_pack_qgrad(const PackT                       &p,
                                                   const MeshT                       &d,
                                                   const ptrdiff_t                    pack,
                                                   const ptrdiff_t                    n_contiguous,
                                                   const ptrdiff_t                    n_ghost,
                                                   const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                                   scalar_t *const SFEM_RESTRICT      pack_qgx,
                                                   scalar_t *const SFEM_RESTRICT      pack_qgy,
                                                   scalar_t *const SFEM_RESTRICT      pack_qgz) {
    const ptrdiff_t owned = p.owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        const ptrdiff_t g = owned + k;
        pack_qgx[k]       = d.qgx[(size_t)g];
        pack_qgy[k]       = d.qgy[(size_t)g];
        pack_qgz[k]       = d.qgz[(size_t)g];
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        const smesh::idx_t g       = ghosts[k];
        pack_qgx[n_contiguous + k] = d.qgx[(size_t)g];
        pack_qgy[n_contiguous + k] = d.qgy[(size_t)g];
        pack_qgz[n_contiguous + k] = d.qgz[(size_t)g];
    }
}

// ---------------------------------------------------------------- nodal pressure gradient
//
// Volume-weighted nodal average of the element-wise gradient of a strided scalar field.
// This is the Rhie-Chow input: the term needs grad p at the nodes, and reconstructing it
// costs a full element sweep -- about 39% of an apply, measured (see
// src/op/cvfem_hex8_ns_op.hpp). Whether it is rebuilt per apply or hoisted out of a Krylov
// solve is therefore a real question and not an implementation detail, which is why both
// callers time it as its own phase rather than folding it into the apply.
//
// Shared because the benchmark and the solver need exactly the same reconstruction, and a
// second copy would be a place for the two to drift on a quantity both of them then feed
// into the same element kernels.
//
// Two constraints shape the signature, both from where this header sits in the include
// order. It is pulled in before either family defines atomic_add, gather_element_coords or
// GeomKind, so the atomic accumulate and the coordinate gather are written out here and
// the geometry is selected by a plain `isoparam` int -- the same convention
// boundary_scs_add_residual already uses. Everything else it calls is either dependent on
// MeshT (and so looked up at instantiation) or comes from the kernel headers above.
// The denominator of that average, 1 / sum_e |det J_e| at each node.
//
// It is pure geometry and constant for the whole solve, and it was being rebuilt inside
// every reconstruction: a fresh nnodes-sized heap allocation, a zero fill, one atomic per
// node per element -- an eighth of the sweep's 32 atomics per element -- and a read of the
// result in the normalisation pass. All of that on the critical path of the largest pass in
// the matvec, for a number that cannot change unless the mesh moves.
//
// Built serially rather than with atomics, deliberately. It is a setup cost paid once, and
// a serial accumulation is reproducible where the atomic one is not, so the reconstruction
// stops inheriting run-to-run variation in its last bits from a quantity that has no reason
// to vary at all.
template <typename MeshT>
static void cvfem_hex8_build_grad_weight(MeshT &d, const int isoparam) {
    if ((ptrdiff_t)d.grad_w_inv.size() == d.nnodes && d.grad_w_isoparam == isoparam &&
        d.grad_w_nelements == d.nelements)
        return;
    d.grad_w_inv.assign((size_t)d.nnodes, scalar_t(0));
    scalar_t *const SFEM_RESTRICT w = d.grad_w_inv.data();
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t det;
        if (isoparam) {
            const auto *const px = d.points[0];
            const auto *const py = d.points[1];
            const auto *const pz = d.points[2];
            scalar_t          x[CVFEM_HEX8_N_NODES], y[CVFEM_HEX8_N_NODES], z[CVFEM_HEX8_N_NODES], adj[9];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const smesh::idx_t g = d.elems[a][e];
                x[a]                 = scalar_t(px[g]);
                y[a]                 = scalar_t(py[g]);
                z[a]                 = scalar_t(pz[g]);
            }
            cvfem_hex8_geom_at(x, y, z, scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), adj, &det);
        } else {
            det = d.jacobian_determinant[(size_t)e];
        }
        const scalar_t vol = std::fabs(det);
        // The same skip the sweep makes, so the weight counts exactly the elements that
        // contribute. Without it a degenerate element would be in the denominator and not
        // in the numerator.
        if (vol < scalar_t(1e-30)) continue;
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) w[d.elems[a][e]] += vol;
    }
    for (ptrdiff_t i = 0; i < d.nnodes; ++i)
        w[i] = w[i] > scalar_t(0) ? scalar_t(1) / w[i] : scalar_t(0);
    d.grad_w_isoparam  = isoparam;
    d.grad_w_nelements = d.nelements;
}

template <typename MeshT>
static void cvfem_hex8_assemble_nodal_grad(MeshT                        &d,
                                           const int                     isoparam,
                                           const scalar_t *const SFEM_RESTRICT src,
                                           const int                     stride,
                                           std::vector<scalar_t>        &ogx,
                                           std::vector<scalar_t>        &ogy,
                                           std::vector<scalar_t>        &ogz) {
    cvfem_hex8_build_grad_weight(d, isoparam);
    ogx.assign((size_t)d.nnodes, scalar_t(0));
    ogy.assign((size_t)d.nnodes, scalar_t(0));
    ogz.assign((size_t)d.nnodes, scalar_t(0));

    scalar_t *const SFEM_RESTRICT pgx = ogx.data();
    scalar_t *const SFEM_RESTRICT pgy = ogy.data();
    scalar_t *const SFEM_RESTRICT pgz = ogz.data();
    const scalar_t *const SFEM_RESTRICT pw = d.grad_w_inv.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t f[CVFEM_HEX8_N_NODES], gx, gy, gz;
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) f[a] = src[(ptrdiff_t)d.elems[a][e] * stride];

        if (isoparam) {
            const auto *const px = d.points[0];
            const auto *const py = d.points[1];
            const auto *const pz = d.points[2];
            scalar_t x[CVFEM_HEX8_N_NODES], y[CVFEM_HEX8_N_NODES], z[CVFEM_HEX8_N_NODES];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const smesh::idx_t g = d.elems[a][e];
                x[a]                 = scalar_t(px[g]);
                y[a]                 = scalar_t(py[g]);
                z[a]                 = scalar_t(pz[g]);
            }
            scalar_t dN[CVFEM_HEX8_N_NODES][3], adj[9], det;
            cvfem_hex8_dn_ref(scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), dN);
            cvfem_hex8_geom_at(x, y, z, scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), adj, &det);
            if (std::fabs(det) < scalar_t(1e-30)) continue;
            scalar_t dr = 0, ds = 0, dt = 0;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                dr += f[a] * dN[a][0];
                ds += f[a] * dN[a][1];
                dt += f[a] * dN[a][2];
            }
            // sgn, not 1/det. What this sweep accumulates is |det| times the gradient, and
            // the gradient is A^T d / det -- so the determinant cancels and only its sign
            // survives. One division and three multiplies per element go with it, and the
            // result is one rounding closer to exact.
            cvfem_hex8_pushforward(adj, det > scalar_t(0) ? scalar_t(1) : scalar_t(-1), dr, ds, dt, gx, gy, gz);
        } else {
            scalar_t adj[9], det;
            load_hex8_adj(d, e, adj, &det);
            if (std::fabs(det) < scalar_t(1e-30)) continue;
            // This is cvfem_hex8_grad_scalar spelled out, with the same det cancellation as
            // above. That function lives in cvfem_hex8_boundary_scs.hpp, which both families
            // include *after* this header, so it cannot be called from here -- and it is
            // itself only these two lines over cvfem_hex8_face_diff and
            // cvfem_hex8_pushforward, both of which come from the kernel header above.
            scalar_t dr, ds, dt;
            cvfem_hex8_face_diff(f, dr, ds, dt);
            cvfem_hex8_pushforward(adj, det > scalar_t(0) ? scalar_t(1) : scalar_t(-1), dr, ds, dt, gx, gy, gz);
        }

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t id = d.elems[a][e];
            CVFEM_ATOMIC_ADD(pgx[id], gx);
            CVFEM_ATOMIC_ADD(pgy[id], gy);
            CVFEM_ATOMIC_ADD(pgz[id], gz);
        }
    }

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t inv = pw[i];
        pgx[i] *= inv;
        pgy[i] *= inv;
        pgz[i] *= inv;
    }
}


// ------------------------------------------------- the same reconstruction, over packs
//
// Same operator, same result to round-off, different sweep. The version above walks the
// flat element table and accumulates with `#pragma omp atomic update` -- 24 atomics per
// element into three global arrays that must first be zeroed. This one reuses the
// arrangement the element sweep next to it already uses: stage the field into a per-pack
// buffer, accumulate into a private per-pack buffer with plain `+=`, write the pack's owned
// rows straight out, and reduce only the shared ghost rows afterwards.
//
// Three things follow from that, beyond the atomics:
//
//   * the three global arrays are WRITTEN rather than accumulated, because the packs
//     partition the owned node range -- so their zero fills disappear;
//   * the result is deterministic. The atomic version is not reproducible even against
//     itself: the same input twice differs in the last bits, because the order in which a
//     node's elements reach it is not fixed. Here the summation order is;
//   * the source is read once per pack node instead of once per element-node incidence,
//     which is eight times less, and contiguously for the owned majority -- so a field read
//     with stride 4 out of an interleaved Krylov vector costs what a contiguous one costs.
//
// Scratch slots 5 and 6, which nothing else uses.
template <typename MeshT, typename PackT>
static void cvfem_hex8_assemble_nodal_grad_packed(MeshT                             &d,
                                                  PackT                             &p,
                                                  const int                          isoparam,
                                                  const scalar_t *const SFEM_RESTRICT src,
                                                  const int                          stride,
                                                  std::vector<scalar_t>             &ogx,
                                                  std::vector<scalar_t>             &ogy,
                                                  std::vector<scalar_t>             &ogz) {
    cvfem_hex8_build_grad_weight(d, isoparam);

    // The owned ranges tile [0, nnodes) exactly, so every entry is written below and there
    // is nothing to pre-zero. If that ever stopped holding, a node no pack owns would keep
    // whatever was in the buffer, so it is checked rather than assumed.
    const bool owns_all = p.n_packs > 0 && p.owned_nodes_ptr[0] == 0 && p.owned_nodes_ptr[p.n_packs] == d.nnodes;
    if (owns_all && (ptrdiff_t)ogx.size() == d.nnodes) {
        ogy.resize((size_t)d.nnodes);
        ogz.resize((size_t)d.nnodes);
    } else {
        ogx.assign((size_t)d.nnodes, scalar_t(0));
        ogy.assign((size_t)d.nnodes, scalar_t(0));
        ogz.assign((size_t)d.nnodes, scalar_t(0));
    }

    scalar_t *const SFEM_RESTRICT gx_out = ogx.data();
    scalar_t *const SFEM_RESTRICT gy_out = ogy.data();
    scalar_t *const SFEM_RESTRICT gz_out = ogz.data();
    const ptrdiff_t               node_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_f   = thread_scratch<scalar_t>(5, (size_t)node_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(6, 3 * (size_t)node_n);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts = &p.ghost_idx[p.ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off = p.ghost_ptr[pack];

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) pack_f[k] = src[(owned + k) * stride];
            for (ptrdiff_t k = 0; k < n_ghost; ++k)
                pack_f[n_contiguous + k] = src[(ptrdiff_t)ghosts[k] * stride];
            std::memset(pack_out, 0, (size_t)n_pack_nodes * 3 * sizeof(scalar_t));

            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                scalar_t fe[CVFEM_HEX8_N_NODES], gx, gy, gz;
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) fe[a] = pack_f[p.elems[a][e]];

                if (isoparam) {
                    const auto *const px = d.points[0];
                    const auto *const py = d.points[1];
                    const auto *const pz = d.points[2];
                    scalar_t x[CVFEM_HEX8_N_NODES], y[CVFEM_HEX8_N_NODES], z[CVFEM_HEX8_N_NODES];
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        const smesh::idx_t g = pack_local_to_global(p, pack, n_contiguous, p.elems[a][e]);
                        x[a]                 = scalar_t(px[g]);
                        y[a]                 = scalar_t(py[g]);
                        z[a]                 = scalar_t(pz[g]);
                    }
                    scalar_t dN[CVFEM_HEX8_N_NODES][3], adj[9], det;
                    cvfem_hex8_dn_ref(scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), dN);
                    cvfem_hex8_geom_at(x, y, z, scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), adj, &det);
                    if (std::fabs(det) < scalar_t(1e-30)) continue;
                    scalar_t dr = 0, ds = 0, dt = 0;
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        dr += fe[a] * dN[a][0];
                        ds += fe[a] * dN[a][1];
                        dt += fe[a] * dN[a][2];
                    }
                    cvfem_hex8_pushforward(adj, det > scalar_t(0) ? scalar_t(1) : scalar_t(-1), dr, ds, dt, gx, gy, gz);
                } else {
                    scalar_t adj[9], det;
                    load_hex8_adj(d, e, adj, &det);
                    if (std::fabs(det) < scalar_t(1e-30)) continue;
                    scalar_t dr, ds, dt;
                    cvfem_hex8_face_diff(fe, dr, ds, dt);
                    cvfem_hex8_pushforward(adj, det > scalar_t(0) ? scalar_t(1) : scalar_t(-1), dr, ds, dt, gx, gy, gz);
                }

                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    scalar_t *const SFEM_RESTRICT o = pack_out + (ptrdiff_t)p.elems[a][e] * 3;
                    o[0] += gx;
                    o[1] += gy;
                    o[2] += gz;
                }
            }

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                gx_out[owned + k] = pack_out[k * 3 + 0];
                gy_out[owned + k] = pack_out[k * 3 + 1];
                gz_out[owned + k] = pack_out[k * 3 + 2];
            }
            scalar_t *const SFEM_RESTRICT bx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT by = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT bz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT o = pack_out + (n_contiguous + k) * 3;
                bx[ghost_off + k]                     = o[0];
                by[ghost_off + k]                     = o[1];
                bz[ghost_off + k]                     = o[2];
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        scalar_t           sx = 0, sy = 0, sz = 0;
        const scalar_t *const SFEM_RESTRICT bx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
        const scalar_t *const SFEM_RESTRICT by = p.ghost_buf.data() + 1 * p.n_ghost_entries;
        const scalar_t *const SFEM_RESTRICT bz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
        for (ptrdiff_t j = begin; j < end; ++j) {
            const ptrdiff_t idx = p.ghost_reduce_idx[j];
            sx += bx[idx];
            sy += by[idx];
            sz += bz[idx];
        }
        gx_out[dest] += sx;
        gy_out[dest] += sy;
        gz_out[dest] += sz;
    }

    const scalar_t *const SFEM_RESTRICT w = d.grad_w_inv.data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        gx_out[i] *= w[i];
        gy_out[i] *= w[i];
        gz_out[i] *= w[i];
    }
}

#endif  // CVFEM_HEX8_PACK_HELPERS_HPP
