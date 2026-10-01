#ifndef CVFEM_HEX8_PACK_HELPERS_HPP
#define CVFEM_HEX8_PACK_HELPERS_HPP

// Geometry and pack-staging helpers shared by the two packed implementations.
//
// The benchmark (cvfem_hex8_best_*.hpp) and the Newton solver
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
                rc.pgx[a][lane]      = pack_pgx[loc];
                rc.pgy[a][lane]      = pack_pgy[loc];
                rc.pgz[a][lane]      = pack_pgz[loc];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
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
// The Rhie-Chow time scale's configuration, resolved once per solve.
//
// One definition for every path -- flat, packed, semi-structured, benchmark -- so none of
// them can end up evaluating a different time scale than the others. Each caller supplies
// a0/dt from its own transient history, which is the only part that differs.
//
// SFEM_RC_TAU=0 is the zero-severity control for the change: no advective branch, no
// transient branch, and twice the scale, which reproduces the previous diffusion-only
// coefficient Df = rc_scale h^2 / (2 mu) exactly. It is a measurement escape hatch, not a
// supported mode -- see cvfem_hex8_rhie_chow_mdot_coeff for why that coefficient was wrong.
struct Hex8RcConfig {
    Hex8RcTau tau;
    scalar_t  scale{0};
};


inline Hex8RcConfig cvfem_hex8_rc_config(const scalar_t rhie_chow_scale, const scalar_t a0_over_dt) {
    // std::getenv rather than smesh::Env, because the benchmark shares this header and is
    // driven by flags with no Env to read through -- the same reason cvfem_env_flag exists.
    // Unset means combined; only an explicit 0 selects the old coefficient.
    static const char *const raw      = std::getenv("SFEM_RC_TAU");
    static const int         combined = !(raw && raw[0] == '0');
    // The transient branch is OFF by default, and that is a measured decision rather than an
    // oversight. SFEM_RC_TAU_TRANSIENT=1 turns it on.
    //
    // V/a_P does carry rho V a0/dt, so on the derivation alone the branch belongs here, and
    // Nalu uses exactly it -- projTimeScale_ = dt/gamma1 -- as its whole time scale. But Nalu
    // is a projection scheme, where dt/gamma1 is the right scaling for the pressure Poisson
    // solve. Ours is a monolithic Newton formulation, and there the branch only shrinks the
    // pressure block towards the unstabilised saddle point as dt falls, which is the failure
    // the "allowing small time steps" line of Rhie-Chow papers exists to address.
    //
    // Measured on the pump at 2,916 dof, one step of dt = 0.1, BDF2, direct preconditioner:
    // the old diffusion-only coefficient does not converge at all (highest Re solved 0 of 20,
    // 21 Newton steps in one stage); advective + diffusive reaches the target in 20 Newton
    // steps over four stages, quadratic in every one of them; adding the transient branch
    // sends the continuation back to bisecting and past 216 Newton steps without converging.
    // The advective branch is the fix and the transient branch is the regression, and they
    // are separable exactly because this flag exists.
    //
    // What is left is Nalu's steady time scale -- "a combined elemental advection and
    // diffusion time scale based on element length along with advection and diffusional
    // parameters" -- which is the one our formulation wants.
    static const char *const rawt     = std::getenv("SFEM_RC_TAU_TRANSIENT");
    static const int         want_dt  = rawt && rawt[0] != '0';
    Hex8RcConfig     c;
    c.tau.u2_scale  = combined ? scalar_t(1) : scalar_t(0);
    c.tau.inv_dt_a0 = (combined && want_dt) ? a0_over_dt : scalar_t(0);
    c.scale         = combined ? rhie_chow_scale : scalar_t(2) * rhie_chow_scale;
    return c;
}

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
        d.rc_coeff.clear();
        d.rc_w.clear();
        return;
    }
    // Rebuilt only when something it depends on moves. rho and mu do move -- the Reynolds
    // continuation walks mu down between stages -- so this cannot be built once at setup
    // and forgotten, and it must not be rebuilt on every matvec either. The state moves too
    // now that the time scale carries the advecting velocity, which is what state_stamp
    // tracks; it changes once per Newton step, not once per matvec.
    // rc_w is validated BESIDE rc_coeff, not assumed to follow it. The two are filled together
    // and read together, so a guard that clears this early on the strength of rc_coeff alone
    // would let the gather read a stale or undersized weight table -- silently, because the
    // weight only scales a Jacobian term and a wrong value looks like a bad linearisation rather
    // than like garbage.
    if (!d.rc_coeff.empty() && d.rc_coeff_rho == rho && d.rc_coeff_mu == mu &&
        d.rc_coeff_scale == d.rhie_chow_scale && d.rc_coeff_stamp == d.state_stamp &&
        (ptrdiff_t)d.rc_coeff.size() == d.nelements * CVFEM_HEX8_N_SCS &&
        (ptrdiff_t)d.rc_w.size() == d.nelements * CVFEM_HEX8_N_SCS)
        return;
    d.rc_coeff.resize((size_t)d.nelements * CVFEM_HEX8_N_SCS);
    d.rc_w.resize((size_t)d.nelements * CVFEM_HEX8_N_SCS);
    d.rc_coeff_rho   = rho;
    d.rc_coeff_mu    = mu;
    d.rc_coeff_scale = d.rhie_chow_scale;
    d.rc_coeff_stamp = d.state_stamp;

    // The two state-dependent inputs to the time scale, resolved once here rather than per
    // element. bdf_coeffs is found by argument-dependent lookup at instantiation -- both
    // MeshData variants declare their own alongside their own transient history, and this
    // template is only ever instantiated where one of them is in scope.
    //
    // SFEM_RC_TAU=0 restores the previous diffusion-only coefficient exactly: no velocity,
    // no transient branch, and twice the scale, because this form's diffusive limit is
    // h^2/(4 nu) against the old h^2/(2 nu). It is the zero-severity control for the
    // change, not a supported mode.
    // Through the per-MeshData resolver rather than a second copy of the rule: both variants
    // define one, and argument-dependent lookup picks the right one at instantiation.
    const Hex8RcConfig cfg = cvfem_hex8_rc_config_for(d);
    const scalar_t rc_scale  = cfg.scale;
    const scalar_t inv_dt_a0 = cfg.tau.inv_dt_a0;
    const scalar_t u2_scale  = cfg.tau.u2_scale;
    const scalar_t *const SFEM_RESTRICT vx = d.ux.data();
    const scalar_t *const SFEM_RESTRICT vy = d.uy.data();
    const scalar_t *const SFEM_RESTRICT vz = d.uz.data();

    const auto *const px = d.points[0];
    const auto *const py = d.points[1];
    const auto *const pz = d.points[2];

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t adj[9], det, A[3][3];
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_dir_areas(adj, A);
        // The three affine edge vectors, from the same adjugate, so this table and the face loops
        // that read it use ONE geometry. Differencing node coordinates here while the face uses
        // the Jacobian column would put the inconsistency back, one level further out.
        scalar_t ecol[3][3];
        cvfem_hex8_affine_edge_cols(adj[0], adj[1], adj[2], adj[3], adj[4], adj[5], adj[6], adj[7], adj[8], det,
                                    ecol[0], ecol[1], ecol[2]);
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
            const scalar_t vax = scalar_t(0.5) * (vx[gi] + vx[gj]);
            const scalar_t vay = scalar_t(0.5) * (vy[gi] + vy[gj]);
            const scalar_t vaz = scalar_t(0.5) * (vz[gi] + vz[gj]);
            const scalar_t u2  = u2_scale * (vax * vax + vay * vay + vaz * vaz);
            const scalar_t dx = ecol[0][q];
            const scalar_t dy = ecol[1][q];
            const scalar_t dz = ecol[2][q];
            d.rc_coeff[(size_t)e * CVFEM_HEX8_N_SCS + s] = cvfem_hex8_rhie_chow_mdot_coeff(rho, mu, rc_scale, dx, dy, dz,
                                                                       A[q][0], A[q][1], A[q][2], u2, inv_dt_a0);
            // The coefficient's velocity-sensitivity weight. It is PURELY geometric plus the two
            // uniforms -- 4*u2_scale*(A.d)^2 / (d.d * scale^2 * (A.A)^2) -- so the Jacobian face
            // loop was recomputing three dot products and a DIVISION per sub-control surface for a
            // number that never varies with the state. Tabulated here, the face reads one value.
            d.rc_w[(size_t)e * CVFEM_HEX8_N_SCS + s] = cvfem_hex8_rhie_chow_du_weight(
                    rc_scale, A[q][0] * A[q][0] + A[q][1] * A[q][1] + A[q][2] * A[q][2],
                    A[q][0] * dx + A[q][1] * dy + A[q][2] * dz, dx * dx + dy * dy + dz * dz, u2_scale);
        }
    }
}

// ------------------------------------------------------- the partially assembled tangent
//
// Sixty scalars per element -- five per sub-control surface -- carrying the whole dependence
// of the Jacobian action on the Newton iterate. Built once per Newton step, read by every
// matvec of the Krylov solve that follows.
//
// This mirrors the state half of cvfem_hex8_conv_face_jv_simd exactly, and mirroring is the
// risk: if that kernel's mass flux, Rhie-Chow correction or upwind switch changes and this
// does not, the two disagree silently and the operator is wrong rather than slow. What
// stops that is not discipline but a test -- cvfem_pa_tangent_test compares the partially
// assembled apply against the direct one, so any divergence in these expressions shows up
// as a failed agreement check rather than as a bad Newton rate months later.
//
// Layout matches d.rc_coeff's: SoA by (surface, component), so the gather is a strided read
// and not a stride-60 walk.
static SFEM_INLINE size_t cvfem_hex8_pa_offset(const ptrdiff_t nelements, const int s, const int c) {
    return (size_t)(s * CVFEM_HEX8_PA_PER_SCS + c) * (size_t)nelements;
}

template <typename MeshT>
static void cvfem_hex8_build_pa_tangent(MeshT &d, const scalar_t rho, const scalar_t mu, const scalar_t ueps) {
    if (d.pa_valid && d.pa_nelements == d.nelements && d.pa_rho == rho && d.pa_mu == mu &&
        d.pa_scale == d.rhie_chow_scale && d.pa_ueps == ueps)
        return;
    // Padded by one SIMD group: the face loops read this store directly, so the lanes of a
    // final group that runs past the end of the mesh read into the pad. Their results are
    // discarded by the scatter, which writes only lanes below nlanes.
    d.pa_tangent.assign((size_t)CVFEM_HEX8_PA_PER_ELEM * (size_t)d.nelements + CVFEM_HEX8_VEC_SIZE,
                        scalar_t(0));
    d.pa_rho       = rho;
    d.pa_mu        = mu;
    d.pa_scale     = d.rhie_chow_scale;
    d.pa_ueps      = ueps;
    d.pa_nelements = d.nelements;
    d.pa_valid     = true;

    const int with_rc = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);
    // The hoisted coefficient, not a fresh evaluation: the face loops read that table and
    // the tangent has to be built from the same numbers they would have used.
    if (with_rc) cvfem_hex8_build_rc_coeff(d, rho, mu);
    const Hex8RcConfig rc_cfg = cvfem_hex8_rc_config_for(d);

    const scalar_t                half = scalar_t(0.5);
    const scalar_t                one  = scalar_t(1);
    scalar_t *const SFEM_RESTRICT out  = d.pa_tangent.data();
    const auto *const             px   = d.points[0];
    const auto *const             py   = d.points[1];
    const auto *const             pz   = d.points[2];

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t adj[9], det, A[3][3];
        load_hex8_adj(d, e, adj, &det);
        cvfem_hex8_dir_areas(adj, A);
        // The same affine edge vector the PA face kernel reads. Building this table from node
        // coordinates while the kernel used the Jacobian column is what cvfem_pa_tangent_test
        // caught at 1.374e-05: the stored tangent and the kernel reading it must agree about the
        // geometry, and on this path the geometry is the Jacobian.
        scalar_t ecol[3][3];
        cvfem_hex8_affine_edge_cols(adj[0], adj[1], adj[2], adj[3], adj[4], adj[5], adj[6], adj[7], adj[8], det,
                                    ecol[0], ecol[1], ecol[2]);

        for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
            const int          i  = CVFEM_HEX8_SCS[s].i;
            const int          j  = CVFEM_HEX8_SCS[s].j;
            const int          q  = s >> 2;
            const smesh::idx_t gi = d.elems[i][e];
            const smesh::idx_t gj = d.elems[j][e];
            const scalar_t     ax = A[q][0], ay = A[q][1], az = A[q][2];

            const scalar_t uxi = d.ux[(size_t)gi], uxj = d.ux[(size_t)gj];
            const scalar_t uyi = d.uy[(size_t)gi], uyj = d.uy[(size_t)gj];
            const scalar_t uzi = d.uz[(size_t)gi], uzj = d.uz[(size_t)gj];

            scalar_t mdot = rho * (half * (uxi + uxj) * ax + half * (uyi + uyj) * ay + half * (uzi + uzj) * az);
            if (with_rc) {
                const scalar_t dx    = ecol[0][q];
                const scalar_t dy    = ecol[1][q];
                const scalar_t dz    = ecol[2][q];
                const scalar_t coeff = d.rc_coeff[(size_t)e * CVFEM_HEX8_N_SCS + s];
                const scalar_t corr  = (d.p[(size_t)gj] - d.p[(size_t)gi]) -
                                      (half * (d.pgx[(size_t)gi] + d.pgx[(size_t)gj]) * dx +
                                       half * (d.pgy[(size_t)gi] + d.pgy[(size_t)gj]) * dy +
                                       half * (d.pgz[(size_t)gi] + d.pgz[(size_t)gj]) * dz);
                mdot -= coeff * corr;
            }

            scalar_t amdot, sgn;
            cvfem_upwind_abs(mdot, ueps, amdot, sgn);
            const scalar_t d_pos = half * (one + sgn);
            const scalar_t d_neg = half * (one - sgn);

            out[cvfem_hex8_pa_offset(d.nelements, s, 0) + (size_t)e] = half * (mdot + amdot);
            out[cvfem_hex8_pa_offset(d.nelements, s, 1) + (size_t)e] = half * (mdot - amdot);
            out[cvfem_hex8_pa_offset(d.nelements, s, 2) + (size_t)e] = d_pos * uxi + d_neg * uxj;
            out[cvfem_hex8_pa_offset(d.nelements, s, 3) + (size_t)e] = d_pos * uyi + d_neg * uyj;
            out[cvfem_hex8_pa_offset(d.nelements, s, 4) + (size_t)e] = d_pos * uzi + d_neg * uzj;
            if (with_rc) {
                // k = g * corr * ubar: the Rhie-Chow coefficient's velocity dependence, which the
                // face loop dots with the direction's face-average velocity.
                const scalar_t dx    = ecol[0][q];
                const scalar_t dy    = ecol[1][q];
                const scalar_t dz    = ecol[2][q];
                const scalar_t corr  = (d.p[(size_t)gj] - d.p[(size_t)gi]) -
                                      (half * (d.pgx[(size_t)gi] + d.pgx[(size_t)gj]) * dx +
                                       half * (d.pgy[(size_t)gi] + d.pgy[(size_t)gj]) * dy +
                                       half * (d.pgz[(size_t)gi] + d.pgz[(size_t)gj]) * dz);
                const scalar_t coeff = d.rc_coeff[(size_t)e * CVFEM_HEX8_N_SCS + s];
                const scalar_t gk    = coeff != scalar_t(0)
                                               ? cvfem_hex8_rhie_chow_coeff_du(
                                                         coeff, cvfem_hex8_rhie_chow_du_weight(
                                                                        rc_cfg.scale, ax * ax + ay * ay + az * az,
                                                                        ax * dx + ay * dy + az * dz,
                                                                        dx * dx + dy * dy + dz * dz, rc_cfg.tau.u2_scale)) *
                                                         corr
                                               : scalar_t(0);
                out[cvfem_hex8_pa_offset(d.nelements, s, 5) + (size_t)e] = gk * half * (uxi + uxj);
                out[cvfem_hex8_pa_offset(d.nelements, s, 6) + (size_t)e] = gk * half * (uyi + uyj);
                out[cvfem_hex8_pa_offset(d.nelements, s, 7) + (size_t)e] = gk * half * (uzi + uzj);
            }
        }
    }
}


// Bytes the store costs, so a speedup can be reported next to its price rather than on its
// own. The assembled matrix is about 847 bytes per degree of freedom for comparison.
template <typename MeshT>
static SFEM_INLINE double cvfem_hex8_pa_bytes_per_dof(const MeshT &d) {
    const double ndof = double(d.nnodes) * double(N_FIELDS);
    return ndof > 0 ? double(d.pa_tangent.size()) * double(sizeof(scalar_t)) / ndof : 0.0;
}

// The SoA gather for the above, straight into the pack the face loops read.
template <typename MeshT>
static SFEM_INLINE void cvfem_hex8_gather_rc_coeff(const MeshT      &d,
                                                   const ptrdiff_t   begin,
                                                   const int         nlanes,
                                                   Hex8RhieChowPack &rc) {
    // Lane-major out of an element-major table: one element's twelve coefficients are
    // consecutive, so this walks one stream instead of twelve. Measured neutral, not faster.
    const scalar_t *const SFEM_RESTRICT src  = d.rc_coeff.data();
    const scalar_t *const SFEM_RESTRICT srcw = d.rc_w.data();
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const scalar_t *const SFEM_RESTRICT e = src + (ptrdiff_t)(begin + lane) * CVFEM_HEX8_N_SCS;
            const scalar_t *const SFEM_RESTRICT w = srcw + (ptrdiff_t)(begin + lane) * CVFEM_HEX8_N_SCS;
            for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
                rc.coeff[s][lane] = e[s];
                rc.wdu[s][lane]   = w[s];
            }
        } else {
            for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
                rc.coeff[s][lane] = scalar_t(0);
                rc.wdu[s][lane]   = scalar_t(0);
            }
        }
    }
    // What the coefficient table was built with, for its velocity sensitivity in the face loop.
    const Hex8RcConfig cfg = cvfem_hex8_rc_config_for(d);
    rc.scale               = cfg.scale;
    rc.tau                 = cfg.tau;
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


// ------------------------------------------------- the same reconstruction, over packs
//
// Same operator, same result to round-off, different sweep. The flat version below walks
// the flat element table and accumulates with `#pragma omp atomic update` -- 24 atomics per
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
// Scratch slots 7 and 8, which nothing else uses.
// All nine components in one FLAT sweep, for the path with no pack to sweep.
//
// The same fusion as the packed version below, and worth doing for a different balance of
// reasons. The atomic count does not change -- seventy-two per element either way, nine per
// node instead of three done three times -- but the adjugate load, the determinant check and
// the d.elems[a][e] indirection drop to a third, the three zeroing passes become one, the
// weight pass runs once instead of three times, and the transpose into the interleaved array
// disappears because this writes there directly.
// WHERE the result goes is a parameter, not a second kernel.
//
// The callers want it in two shapes: interleaved, nc components per node in one buffer, which
// is what the deferred correction and the fused sets read; and one array per component, which
// is what the pressure gradient's consumers -- every element kernel, generated and hand
// written -- read as pgx/pgy/pgz. Both are "a base pointer per component plus a stride shared
// by all of them": interleaved is base = out + c with stride nc, split is base = og_c with
// stride 1. So the sweep takes `outp` and `out_stride` and serves both, where before there
// were two copies of it differing in nothing else.
//
// The input field's stride is per field for the same reason. The exact higher-order Jacobian
// action reconstructs the gradient of the Krylov direction, which is interleaved four-wide;
// with a unit-stride-only sweep it had to de-interleave three components into scratch on every
// matvec, and that pass is now gone.
// ISO is compile time beside NC, for the same reason the limiter is on the Jacobian action: it is
// SWEEP-UNIFORM, so a runtime test of it inside the element loop asks the same question of every
// element and blocks the vectoriser from either body. It also lets the reference shape derivatives
// be hoisted: they are evaluated at the fixed element centre, so `dN` does not depend on the
// element at all and computing it per element was work the affine path did not even want.
// The geometry kind stays a RUNTIME argument, unlike NC beside it, and that is a measured
// decision rather than an oversight. Making it a template parameter -- with `if constexpr` on both
// branches and the reference table hoisted -- is the tidier shape and cost the velocity
// reconstruction 4654 -> 4243 MDOF/s on Grace at n=128, reproduced on three nodes, while leaving
// the pressure arm unchanged. It was tried with the table inside the parallel region as well, at
// 4172. The branch is perfectly predicted and the ISO=0 body is what runs; whatever the extra
// instantiations do to inlining or register allocation in the nine-component sweep costs more than
// the branch does. Revisit only with a measurement.
template <int NC, typename MeshT>
static void cvfem_hex8_nodal_grads_atomic_nc(MeshT                               &d,
                                             const int                            isoparam,
                                             const scalar_t *const SFEM_RESTRICT *srcs,
                                             const int *const                     src_stride,
                                             scalar_t *const *const               outp,
                                             const ptrdiff_t *const               out_stride) {
    constexpr int nc = NC;
    constexpr int nf = NC / 3;
    cvfem_hex8_build_grad_weight(d, isoparam);

    const scalar_t *const SFEM_RESTRICT pw = d.grad_w_inv.data();

    // Everything is zeroed because the accumulation below is additive.
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i)
        for (int c = 0; c < nc; ++c) outp[c][i * out_stride[c]] = scalar_t(0);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        smesh::idx_t id[CVFEM_HEX8_N_NODES];
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) id[a] = d.elems[a][e];

        scalar_t adj[9], det, dN[CVFEM_HEX8_N_NODES][3];
        if (isoparam) {
            const auto *const px = d.points[0];
            const auto *const py = d.points[1];
            const auto *const pz = d.points[2];
            scalar_t x[CVFEM_HEX8_N_NODES], y[CVFEM_HEX8_N_NODES], z[CVFEM_HEX8_N_NODES];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                x[a] = scalar_t(px[id[a]]);
                y[a] = scalar_t(py[id[a]]);
                z[a] = scalar_t(pz[id[a]]);
            }
            cvfem_hex8_dn_ref(scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), dN);
            cvfem_hex8_geom_at(x, y, z, scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), adj, &det);
        } else {
            load_hex8_adj(d, e, adj, &det);
        }
        if (std::fabs(det) < scalar_t(1e-30)) continue;
        const scalar_t sgn = det > scalar_t(0) ? scalar_t(1) : scalar_t(-1);

        scalar_t g[nc];
        for (int r = 0; r < nf; ++r) {
            const ptrdiff_t st = src_stride ? (ptrdiff_t)src_stride[r] : (ptrdiff_t)1;
            scalar_t        f[CVFEM_HEX8_N_NODES];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) f[a] = srcs[r][(ptrdiff_t)id[a] * st];
            scalar_t dr, ds, dt;
            if (isoparam) {
                dr = ds = dt = 0;
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    dr += f[a] * dN[a][0];
                    ds += f[a] * dN[a][1];
                    dt += f[a] * dN[a][2];
                }
            } else {
                cvfem_hex8_face_diff(f, dr, ds, dt);
            }
            cvfem_hex8_pushforward(adj, sgn, dr, ds, dt, g[r * 3 + 0], g[r * 3 + 1], g[r * 3 + 2]);
        }

        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            for (int c = 0; c < nc; ++c) CVFEM_ATOMIC_ADD(outp[c][(ptrdiff_t)id[a] * out_stride[c]], g[c]);
        }
    }

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
        const scalar_t inv = pw[i];
        for (int c = 0; c < nc; ++c) outp[c][i * out_stride[c]] *= inv;
    }
}

// HOW MANY components is compile time, so the scatter's trip count is.
//
// The element gradient is broadcast to all eight of the element's nodes, which is nc adds per
// node and the hottest loop in the sweep. With nc a runtime value that loop keeps a counter and
// cannot be unrolled, and the pressure gradient -- three components, the shortest trip -- pays
// most for it: unifying the two sweeps cost it 5.7% on Grace until the count moved into the
// template. The same lesson the limiter select in the Jacobian action taught, in the same place.
//
// Four fields is the widest set any caller asks for: the pressure and the three velocity
// components together. The count is checked here and not inside the sweep.
template <typename MeshT>
static void cvfem_hex8_assemble_nodal_grads_atomic(MeshT                               &d,
                                                   const int                            isoparam,
                                                   const scalar_t *const SFEM_RESTRICT *srcs,
                                                   const int *const                     src_stride,
                                                   const int                            nf,
                                                   scalar_t *const *const               outp,
                                                   const ptrdiff_t *const               out_stride) {
    switch (nf) {
        case 1: cvfem_hex8_nodal_grads_atomic_nc<3>(d, isoparam, srcs, src_stride, outp, out_stride); break;
        case 2: cvfem_hex8_nodal_grads_atomic_nc<6>(d, isoparam, srcs, src_stride, outp, out_stride); break;
        case 3: cvfem_hex8_nodal_grads_atomic_nc<9>(d, isoparam, srcs, src_stride, outp, out_stride); break;
        case 4: cvfem_hex8_nodal_grads_atomic_nc<12>(d, isoparam, srcs, src_stride, outp, out_stride); break;
        default: assert(false && "nodal gradient sweep takes 1 to 4 fields"); break;
    }
}

// Interleaved output into a vector the sweep grows for the caller. Grown, never shrunk: the
// caller alternates component counts between evaluations (a frozen residual asks for p alone,
// the two that build the held correction ask for p and u together), and reassigning would
// reallocate on every one of them.
template <typename MeshT>
static void cvfem_hex8_assemble_nodal_grads_atomic(MeshT                               &d,
                                                   const int                            isoparam,
                                                   const scalar_t *const SFEM_RESTRICT *srcs,
                                                   const int                            nf,
                                                   std::vector<scalar_t>               &out,
                                                   const int *const                     src_stride = nullptr) {
    const int nc = 3 * nf;
    if ((ptrdiff_t)out.size() < d.nnodes * nc) out.resize((size_t)d.nnodes * nc);
    scalar_t  *outp[12];
    ptrdiff_t  ostr[12];
    for (int c = 0; c < nc; ++c) { outp[c] = out.data() + c; ostr[c] = nc; }
    cvfem_hex8_assemble_nodal_grads_atomic(d, isoparam, srcs, src_stride, nf, outp, ostr);
}

// One strided field into three component arrays -- the pressure gradient, and the Krylov
// direction's pressure component inside the Jacobian action.
template <typename MeshT>
static void cvfem_hex8_assemble_nodal_grads_atomic(MeshT                              &d,
                                                   const int                           isoparam,
                                                   const scalar_t *const SFEM_RESTRICT src,
                                                   const int                           stride,
                                                   std::vector<scalar_t>              &ogx,
                                                   std::vector<scalar_t>              &ogy,
                                                   std::vector<scalar_t>              &ogz) {
    ogx.resize((size_t)d.nnodes);
    ogy.resize((size_t)d.nnodes);
    ogz.resize((size_t)d.nnodes);
    const scalar_t *const SFEM_RESTRICT srcs[1] = {src};
    const int                           st[1]   = {stride};
    scalar_t                           *outp[3] = {ogx.data(), ogy.data(), ogz.data()};
    const ptrdiff_t                     ostr[3] = {1, 1, 1};
    cvfem_hex8_assemble_nodal_grads_atomic(d, isoparam, srcs, st, 1, outp, ostr);
}


// All NINE components of the nodal velocity gradient in ONE pack sweep.
//
// assemble_nodal_u_grad used to call the single-field sweep below three times, once per
// velocity component. Everything that is not the field itself was therefore done three
// times: the adjugate load, the determinant check, the p.elems[a][e] indirection (eight per
// element, twice per sweep, so forty-eight instead of sixteen), the pack_out memset, the
// weight pass and the whole ghost reduction. Only the reference gradient and the pushforward
// are genuinely per-component.
//
// The shape is taken from the fast kernels: gather once, geometry once, then the per-field
// arithmetic inside, which is what the sum-factorised residual does with its element data.
//
// It keeps its own ghost buffer because PackedData::ghost_buf is sized N_FIELDS (four) wide
// and this needs nine. Held by the caller and reused, so a residual does not allocate.
template <int NC, typename MeshT, typename PackT>
static void cvfem_hex8_nodal_grads_packed_nc(MeshT                               &d,
                                             PackT                               &p,
                                             const int                            isoparam,
                                             const scalar_t *const SFEM_RESTRICT *srcs,
                                             const int *const                     src_stride,
                                             scalar_t *const *const               outp,
                                             const ptrdiff_t *const               out_stride,
                                             std::vector<scalar_t>               &gbuf) {
    constexpr int nc = NC;
    constexpr int nf = NC / 3;
    cvfem_hex8_build_grad_weight(d, isoparam);

    const bool owns_all = p.n_packs > 0 && p.owned_nodes_ptr[0] == 0 && p.owned_nodes_ptr[p.n_packs] == d.nnodes;
    // The owned ranges tile [0, nnodes) exactly, so every entry is written below and there is
    // nothing to pre-zero. If that ever stopped holding, a node no pack owns would keep
    // whatever was in the buffer, so it is checked rather than assumed.
    if (!owns_all) {
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < d.nnodes; ++i)
            for (int c = 0; c < nc; ++c) outp[c][i * out_stride[c]] = scalar_t(0);
    }
    // The ghost entries are stored, not accumulated, so growing the buffer is all this needs.
    if ((ptrdiff_t)gbuf.size() < (ptrdiff_t)p.n_ghost_entries * nc)
        gbuf.resize((size_t)p.n_ghost_entries * nc, scalar_t(0));

    scalar_t *const SFEM_RESTRICT       gb = gbuf.data();
    const scalar_t *const SFEM_RESTRICT w  = d.grad_w_inv.data();
    const ptrdiff_t node_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_f   = thread_scratch<scalar_t>(7, 4 * (size_t)node_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(8, 12 * (size_t)node_n);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts    = &p.ghost_idx[p.ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off = p.ghost_ptr[pack];

            // One gather for all three fields, component-major so the element read below is
            // three contiguous strides rather than three separate arrays.
            for (int r = 0; r < nf; ++r) {
                const ptrdiff_t               st  = src_stride ? (ptrdiff_t)src_stride[r] : (ptrdiff_t)1;
                scalar_t *const SFEM_RESTRICT dst = pack_f + (ptrdiff_t)r * node_n;
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) dst[k] = srcs[r][(owned + k) * st];
                for (ptrdiff_t k = 0; k < n_ghost; ++k) dst[n_contiguous + k] = srcs[r][(ptrdiff_t)ghosts[k] * st];
            }
            std::memset(pack_out, 0, (size_t)n_pack_nodes * nc * sizeof(scalar_t));

            // THE ELEMENT LOOP IS SCALAR, and stays scalar. Lane-blocking it over
            // CVFEM_HEX8_VEC_SIZE elements -- reusing gather_hex8_adj_soa for the geometry, as
            // every operator sweep beside this one does -- was built and measured: the velocity
            // reconstruction fell 4716 -> 3647 MDOF/s, and 3781 with the geometry branch hoisted
            // out of the vector body, at n=128 on Grace.
            //
            // The reason it loses is that the half a lane loop can vectorise is not the half that
            // costs. The scatter broadcasts the element gradient to all eight nodes and cannot be
            // vectorised at all, because two elements in a batch may share a node; and the
            // per-element work a lane loop does speed up is dominated by the adjugate LOAD, which
            // staging turns into a memcpy without making it cheaper. What lane-blocking adds is a
            // transpose the scalar loop never pays -- 8xVEC_SIZE gathers into fe, plus loc, cof,
            // dr/ds/dt and g held as batches -- to feed a scatter that is still scalar.
            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                int      loc[CVFEM_HEX8_N_NODES];
                scalar_t adj[9], det;
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) loc[a] = p.elems[a][e];

                scalar_t dN[CVFEM_HEX8_N_NODES][3];
                if (isoparam) {
                    const auto *const px = d.points[0];
                    const auto *const py = d.points[1];
                    const auto *const pz = d.points[2];
                    scalar_t x[CVFEM_HEX8_N_NODES], y[CVFEM_HEX8_N_NODES], z[CVFEM_HEX8_N_NODES];
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        const smesh::idx_t g = pack_local_to_global(p, pack, n_contiguous, loc[a]);
                        x[a]                 = scalar_t(px[g]);
                        y[a]                 = scalar_t(py[g]);
                        z[a]                 = scalar_t(pz[g]);
                    }
                    cvfem_hex8_dn_ref(scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), dN);
                    cvfem_hex8_geom_at(x, y, z, scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), adj, &det);
                } else {
                    load_hex8_adj(d, e, adj, &det);
                }
                if (std::fabs(det) < scalar_t(1e-30)) continue;
                const scalar_t sgn = det > scalar_t(0) ? scalar_t(1) : scalar_t(-1);

                scalar_t g[nc];
                for (int r = 0; r < nf; ++r) {
                    const scalar_t *const SFEM_RESTRICT fsrc = pack_f + (ptrdiff_t)r * node_n;
                    scalar_t fe[CVFEM_HEX8_N_NODES];
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) fe[a] = fsrc[loc[a]];
                    scalar_t dr, ds, dt;
                    if (isoparam) {
                        dr = ds = dt = 0;
                        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                            dr += fe[a] * dN[a][0];
                            ds += fe[a] * dN[a][1];
                            dt += fe[a] * dN[a][2];
                        }
                    } else {
                        cvfem_hex8_face_diff(fe, dr, ds, dt);
                    }
                    cvfem_hex8_pushforward(adj, sgn, dr, ds, dt, g[r * 3 + 0], g[r * 3 + 1], g[r * 3 + 2]);
                }

                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    scalar_t *const SFEM_RESTRICT o = pack_out + (ptrdiff_t)loc[a] * nc;
                    for (int k = 0; k < nc; ++k) o[k] += g[k];
                }
            }

            // WHY THE OWNED ROWS ARE STAGED AT ALL, given that no other pack writes them and
            // they could be accumulated straight into the output. That was built and measured:
            // the velocity reconstruction fell 4716 -> 4054 MDOF/s and the pressure one
            // 10268 -> 9695, at n=128 on Grace. Writing through costs a per-node owned/ghost test
            // in the innermost scatter and it breaks the fusion below -- the weight stops being
            // free and becomes its own read-modify-write over the owned window -- and together
            // those outweigh the staging read they remove.
            //
            // The denominator is applied here and again in the ghost reduction rather than in
            // a pass of its own, because the average is linear in its numerator:
            // (owned + ghost) * w is owned * w + ghost * w. That removes a full read-modify-
            // write over the nodal arrays from every matvec.
            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const scalar_t  wi = w[owned + k];
                for (int c = 0; c < nc; ++c) outp[c][(owned + k) * out_stride[c]] = pack_out[k * nc + c] * wi;
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT o = pack_out + (n_contiguous + k) * nc;
                for (int c = 0; c < nc; ++c) gb[(ghost_off + k) * nc + c] = o[c];
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        scalar_t           s[nc] = {0};
        for (ptrdiff_t j = begin; j < end; ++j) {
            const ptrdiff_t idx = p.ghost_reduce_idx[j];
            for (int c = 0; c < nc; ++c) s[c] += gb[idx * nc + c];
        }
        const scalar_t wd = w[dest];
        for (int c = 0; c < nc; ++c) outp[c][(ptrdiff_t)dest * out_stride[c]] += s[c] * wd;
    }
}

// HOW MANY components is compile time, so the scatter's trip count is.
//
// The element gradient is broadcast to all eight of the element's nodes, which is nc adds per
// node and the hottest loop in the sweep. With nc a runtime value that loop keeps a counter and
// cannot be unrolled, and the pressure gradient -- three components, the shortest trip -- pays
// most for it: unifying the two sweeps cost it 5.7% on Grace until the count moved into the
// template. The same lesson the limiter select in the Jacobian action taught, in the same place.
//
// Four fields is the widest set any caller asks for: the pressure and the three velocity
// components together. The count is checked here and not inside the sweep.
template <typename MeshT, typename PackT>
static void cvfem_hex8_assemble_nodal_grads_packed(MeshT                               &d,
                                                   PackT                               &p,
                                                   const int                            isoparam,
                                                   const scalar_t *const SFEM_RESTRICT *srcs,
                                                   const int *const                     src_stride,
                                                   const int                            nf,
                                                   scalar_t *const *const               outp,
                                                   const ptrdiff_t *const               out_stride,
                                                   std::vector<scalar_t>               &gbuf) {
    switch (nf) {
        case 1: cvfem_hex8_nodal_grads_packed_nc<3>(d, p, isoparam, srcs, src_stride, outp, out_stride, gbuf); break;
        case 2: cvfem_hex8_nodal_grads_packed_nc<6>(d, p, isoparam, srcs, src_stride, outp, out_stride, gbuf); break;
        case 3: cvfem_hex8_nodal_grads_packed_nc<9>(d, p, isoparam, srcs, src_stride, outp, out_stride, gbuf); break;
        case 4: cvfem_hex8_nodal_grads_packed_nc<12>(d, p, isoparam, srcs, src_stride, outp, out_stride, gbuf); break;
        default: assert(false && "nodal gradient sweep takes 1 to 4 fields"); break;
    }
}

// Interleaved output into a vector the sweep grows for the caller -- see the flat twin above
// for why it is grown and never shrunk. A buffer wider than nnodes * nc is harmless because
// every index below is i * nc + c.
template <typename MeshT, typename PackT>
static void cvfem_hex8_assemble_nodal_grads_packed(MeshT                               &d,
                                                   PackT                               &p,
                                                   const int                            isoparam,
                                                   const scalar_t *const SFEM_RESTRICT *srcs,
                                                   const int                            nf,
                                                   std::vector<scalar_t>               &out,
                                                   std::vector<scalar_t>               &gbuf,
                                                   const int *const                     src_stride = nullptr) {
    const int nc = 3 * nf;
    if ((ptrdiff_t)out.size() < d.nnodes * nc) out.resize((size_t)d.nnodes * nc);
    scalar_t  *outp[12];
    ptrdiff_t  ostr[12];
    for (int c = 0; c < nc; ++c) { outp[c] = out.data() + c; ostr[c] = nc; }
    cvfem_hex8_assemble_nodal_grads_packed(d, p, isoparam, srcs, src_stride, nf, outp, ostr, gbuf);
}

// One strided field into three component arrays -- the pressure gradient, and the Krylov
// direction's pressure component inside the Jacobian action.
template <typename MeshT, typename PackT>
static void cvfem_hex8_assemble_nodal_grads_packed(MeshT                              &d,
                                                   PackT                              &p,
                                                   const int                           isoparam,
                                                   const scalar_t *const SFEM_RESTRICT src,
                                                   const int                           stride,
                                                   std::vector<scalar_t>              &ogx,
                                                   std::vector<scalar_t>              &ogy,
                                                   std::vector<scalar_t>              &ogz,
                                                   std::vector<scalar_t>              &gbuf) {
    ogx.resize((size_t)d.nnodes);
    ogy.resize((size_t)d.nnodes);
    ogz.resize((size_t)d.nnodes);
    const scalar_t *const SFEM_RESTRICT srcs[1] = {src};
    const int                           st[1]   = {stride};
    scalar_t                           *outp[3] = {ogx.data(), ogy.data(), ogz.data()};
    const ptrdiff_t                     ostr[3] = {1, 1, 1};
    cvfem_hex8_assemble_nodal_grads_packed(d, p, isoparam, srcs, st, 1, outp, ostr, gbuf);
}


#endif  // CVFEM_HEX8_PACK_HELPERS_HPP
