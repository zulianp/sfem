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
template <typename MeshT>
static void cvfem_hex8_assemble_nodal_grad(const MeshT                  &d,
                                           const int                     isoparam,
                                           const scalar_t *const SFEM_RESTRICT src,
                                           const int                     stride,
                                           std::vector<scalar_t>        &ogx,
                                           std::vector<scalar_t>        &ogy,
                                           std::vector<scalar_t>        &ogz) {
    ogx.assign((size_t)d.nnodes, scalar_t(0));
    ogy.assign((size_t)d.nnodes, scalar_t(0));
    ogz.assign((size_t)d.nnodes, scalar_t(0));
    std::vector<scalar_t> w((size_t)d.nnodes, scalar_t(0));

    scalar_t *const SFEM_RESTRICT pgx = ogx.data();
    scalar_t *const SFEM_RESTRICT pgy = ogy.data();
    scalar_t *const SFEM_RESTRICT pgz = ogz.data();
    scalar_t *const SFEM_RESTRICT pw  = w.data();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t f[CVFEM_HEX8_N_NODES], gx, gy, gz, vol;
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
            vol = std::fabs(det);
            if (vol < scalar_t(1e-30)) continue;
            scalar_t dr = 0, ds = 0, dt = 0;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                dr += f[a] * dN[a][0];
                ds += f[a] * dN[a][1];
                dt += f[a] * dN[a][2];
            }
            cvfem_hex8_pushforward(adj, scalar_t(1) / det, dr, ds, dt, gx, gy, gz);
        } else {
            scalar_t adj[9], det;
            load_hex8_adj(d, e, adj, &det);
            vol = std::fabs(det);
            if (vol < scalar_t(1e-30)) continue;
            // This is cvfem_hex8_grad_scalar spelled out. That function lives in
            // cvfem_hex8_boundary_scs.hpp, which both families include *after* this
            // header, so it cannot be called from here -- and it is itself only these two
            // lines over cvfem_hex8_face_diff and cvfem_hex8_pushforward, both of which
            // come from the kernel header above and are in scope.
            scalar_t dr, ds, dt;
            cvfem_hex8_face_diff(f, dr, ds, dt);
            cvfem_hex8_pushforward(adj, scalar_t(1) / det, dr, ds, dt, gx, gy, gz);
        }

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t id = d.elems[a][e];
            CVFEM_ATOMIC_ADD(pgx[id], vol * gx);
            CVFEM_ATOMIC_ADD(pgy[id], vol * gy);
            CVFEM_ATOMIC_ADD(pgz[id], vol * gz);
            CVFEM_ATOMIC_ADD(pw[id], vol);
        }
    }

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        if (pw[i] <= scalar_t(0)) continue;
        const scalar_t inv = scalar_t(1) / pw[i];
        pgx[i] *= inv;
        pgy[i] *= inv;
        pgz[i] *= inv;
    }
}

#endif  // CVFEM_HEX8_PACK_HELPERS_HPP
