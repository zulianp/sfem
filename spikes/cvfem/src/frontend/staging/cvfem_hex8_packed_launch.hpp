#ifndef CVFEM_HEX8_PACKED_LAUNCH_HPP
#define CVFEM_HEX8_PACKED_LAUNCH_HPP

// THE PACKED LAYOUT'S PACK BUILDER AND SIX LAUNCHERS.
//
// Everything here reads a mesh or a pack. build_pack_local_crs computes the pack-local CRS and
// publishes the pointer arrays the assembly kernel indexes. The six launchers resolve the arrays
// out of MeshData and PackedData, run the one-time setup their sweeps used to do themselves
// (zeroing the matrix, building the Rhie-Chow coefficients), own the `#pragma omp parallel`,
// split the packs across threads as a cvfem_range, dispatch on the kernel variant and the
// geometry, and reduce the ghost rows afterwards.
//
// None of that is element arithmetic, and all of it needs the staging objects -- which is why
// kernels/packed/cvfem_hex8_best_packed.hpp included the bench's staging header for as long as
// they lived there. It no longer does.
#include "frontend/staging/cvfem_hex8_best_common.hpp"
#include "kernels/packed/cvfem_hex8_best_packed.hpp"
#include "kernels/packed/affine/cvfem_hex8_best_packed_affine.hpp"
#include "kernels/packed/isoparametric/cvfem_hex8_best_packed_isoparam.hpp"

// The scalar higher-order sweep this header used to define. It is the slower of the packed
// layout's two higher-order kernels (Grace job 4981920) and lives in subpar/; the stub in
// cvfem_hex8_best_common.hpp stands in its place when the flag is off.
#ifdef CVFEM_ENABLE_SUBPAR
#include "cvfem_hex8_packed_defcor_scalar.hpp"
#endif

static void build_pack_local_crs(PackedData               &p,
                                 const ptrdiff_t           nelements,
                                 const count_t     *rowptr_g,
                                 const idx_t       *colidx_g) {
    p.local_rowptr.resize((size_t)p.n_packs);
    p.local_colidx.resize((size_t)p.n_packs);
    p.local_global_slot.resize((size_t)p.n_packs);
    p.local_element_slot.assign((size_t)nelements * CVFEM_HEX8_N_NODES * CVFEM_HEX8_N_NODES, 0);
    p.max_local_nnz = 0;

    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack];
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
        const ptrdiff_t e_start      = pack * p.n_elements_per_pack;
        const ptrdiff_t e_end        = std::min(nelements, (pack + 1) * p.n_elements_per_pack);

        std::vector<std::vector<pack_idx_t>> adj((size_t)n_pack_nodes);
        for (ptrdiff_t e = e_start; e < e_end; ++e) {
            pack_idx_t ev[CVFEM_HEX8_N_NODES];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) ev[a] = p.elems[a][e];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                for (int b = 0; b < CVFEM_HEX8_N_NODES; ++b) adj[(size_t)ev[a]].push_back(ev[b]);
            }
        }

        auto &rowptr = p.local_rowptr[(size_t)pack];
        auto &colidx = p.local_colidx[(size_t)pack];
        rowptr.assign((size_t)n_pack_nodes + 1, 0);
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) {
            auto &row = adj[(size_t)i];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            rowptr[(size_t)i + 1] = (int)row.size();
        }
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) rowptr[(size_t)i + 1] += rowptr[(size_t)i];
        colidx.resize((size_t)rowptr[(size_t)n_pack_nodes]);
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) {
            const auto &row = adj[(size_t)i];
            std::memcpy(colidx.data() + rowptr[(size_t)i], row.data(), row.size() * sizeof(pack_idx_t));
        }

        auto &global_slots = p.local_global_slot[(size_t)pack];
        global_slots.resize(colidx.size());
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) {
            const idx_t grow  = pack_local_to_global(p, pack, n_contiguous, (pack_idx_t)i);
            const int          begin = rowptr[(size_t)i];
            const int          end   = rowptr[(size_t)i + 1];
            for (int t = begin; t < end; ++t) {
                const idx_t gcol = pack_local_to_global(p, pack, n_contiguous, colidx[(size_t)t]);
                global_slots[(size_t)t] = find_bsr_slot(rowptr_g, colidx_g, grow, gcol);
            }
        }
        p.max_local_nnz = std::max(p.max_local_nnz, (ptrdiff_t)colidx.size());

        for (ptrdiff_t e = e_start; e < e_end; ++e) {
            int *const slots = p.local_element_slot.data() + (size_t)e * 64;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t local_row = p.elems[a][e];
                const int        row_begin = rowptr[(size_t)local_row];
                const int        row_len   = rowptr[(size_t)local_row + 1] - row_begin;
                const pack_idx_t *row      = colidx.data() + row_begin;
                for (int b = 0; b < CVFEM_HEX8_N_NODES; ++b) {
                    slots[a * 8 + b] = row_begin + find_pack_col(p.elems[b][e], row, row_len);
                }
            }
        }
    }

    p.ghost_mat_ptr.assign((size_t)p.n_ghost_entries + 1, 0);
    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack];
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t ghost_off    = p.ghost_ptr[pack];
        const auto     &rowptr       = p.local_rowptr[(size_t)pack];
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
            const ptrdiff_t local_i = n_contiguous + k;
            p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k + 1] = rowptr[(size_t)local_i + 1] - rowptr[(size_t)local_i];
        }
    }
    for (ptrdiff_t i = 0; i < p.n_ghost_entries; ++i) p.ghost_mat_ptr[(size_t)i + 1] += p.ghost_mat_ptr[(size_t)i];

    const ptrdiff_t gnnz = p.ghost_mat_ptr[(size_t)p.n_ghost_entries];
    if (getenv("CVFEM_PACK_STATS")) {
        ptrdiff_t sum_local_nnz = 0;
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) sum_local_nnz += (ptrdiff_t)p.local_colidx[(size_t)pack].size();
        std::printf("[pack-stats] sum_local_nnz=%td ghost_nnz=%td\n", sum_local_nnz, gnnz);
    }
    p.ghost_mat_slot.resize((size_t)gnnz);
    p.ghost_mat_val.assign((size_t)gnnz * 16, 0.0);

    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack];
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t ghost_off    = p.ghost_ptr[pack];
        const auto     &rowptr       = p.local_rowptr[(size_t)pack];
        const auto     &colidx       = p.local_colidx[(size_t)pack];
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
            const ptrdiff_t local_i = n_contiguous + k;
            const int       begin   = rowptr[(size_t)local_i];
            const int       end     = rowptr[(size_t)local_i + 1];
            const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
            const idx_t grow = p.ghost_idx[(size_t)ghost_off + (size_t)k];
            for (int t = 0; t < end - begin; ++t) {
                const idx_t gcol = pack_local_to_global(p, pack, n_contiguous, colidx[(size_t)begin + t]);
                p.ghost_mat_slot[(size_t)dest + (size_t)t] = find_bsr_slot(rowptr_g, colidx_g, grow, gcol);
            }
        }
    }

    // Published after the loop above, which is the only thing that sizes them.
    p.local_rowptr_ptr.resize((size_t)p.n_packs);
    p.local_global_slot_ptr.resize((size_t)p.n_packs);
    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        p.local_rowptr_ptr[(size_t)pack]      = p.local_rowptr[(size_t)pack].data();
        p.local_global_slot_ptr[(size_t)pack] = p.local_global_slot[(size_t)pack].data();
    }
}

static SFEM_NOINLINE void apply_residual_packed_defcor(MeshData       &d,
                                                       PackedData     &p,
                                                       const scalar_t  rho,
                                                       const scalar_t  mu,
                                                       const scalar_t *const SFEM_RESTRICT ugrad,
                                                       const int       limiter,
                                                       const scalar_t  venkat_c) {
    scalar_t *const SFEM_RESTRICT rx = d.rx.data();
    scalar_t *const SFEM_RESTRICT ry = d.ry.data();
    scalar_t *const SFEM_RESTRICT rz = d.rz.data();
    scalar_t *const SFEM_RESTRICT rc = d.rc.data();
    const size_t                  scratch_n = packed_scratch_n(p.max_actual_nodes_per_pack);
    const Hex8Extras              opt = cvfem_hex8_extras_of(d);
    const int                     with_rc = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);

#pragma omp parallel
    apply_residual_packed_defcor_range(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.rhie_chow_scale, d.ux.data(), d.uy.data(), d.uz.data(), p.elems, p.ghost_buf.data(), p.ghost_idx, p.ghost_ptr, p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.n_ghost_entries, p.owned_nodes_ptr, rho, mu, ugrad, limiter, venkat_c, rx, ry, rz, rc, scratch_n, with_rc);

    scalar_t *const fields[CVFEM_HEX8_N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (int f = 0; f < CVFEM_HEX8_N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + (ptrdiff_t)f * p.n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            fields[f][dest] += sum;
        }
    }
}

// THE FRONT END DISPATCHES ON GeomKind, which is DESIGN.md's correction: "GeomKind is used at
// the front end level to dispatch based on the type of elements in the block (now smesh also
// provides such enums) and it can be overriden at runtime." The two sweeps are separate
// functions in packed/affine/ and packed/isoparametric/, and this is the only place that
// chooses between them -- nothing below here tests the geometry.
static SFEM_NOINLINE void apply_residual_packed(MeshData        &d,
                                                PackedData      &p,
                                                const scalar_t   rho,
                                                const scalar_t   mu,
                                                const GeomKind   geom) {
    const scalar_t *const SFEM_RESTRICT ux = d.ux.data();
    const scalar_t *const SFEM_RESTRICT uy = d.uy.data();
    const scalar_t *const SFEM_RESTRICT uz = d.uz.data();
    const scalar_t *const SFEM_RESTRICT pr = d.p.data();
    scalar_t *const SFEM_RESTRICT       rx = d.rx.data();
    scalar_t *const SFEM_RESTRICT       ry = d.ry.data();
    scalar_t *const SFEM_RESTRICT       rz = d.rz.data();
    scalar_t *const SFEM_RESTRICT       rc = d.rc.data();
    const size_t                        scratch_n = packed_scratch_n(p.max_actual_nodes_per_pack);
    // Rhie-Chow needs the coordinates and the nodal gradient staged per pack, six arrays
    // rather than three, so slot 3 is sized for six when it is on. This is the solver's
    // own arrangement (packed_rc_n, cvfem_hex8_ns_packed.hpp) and the constant already
    // lives in the shared cvfem_hex8_pack_common.hpp.
    const int                           with_rc   = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);


// ONE ARM PER GEOMETRY. The sweep was also instantiated once per micro-kernel variant,
    // chosen by a switch here; DESIGN.md's correction leaves one micro-kernel per kernel, and on
    // Grace at 28,756k dof the affine one measured 639 MELEM/s against 518 and 505 for the two
    // arms it replaced (perf/campaign_generated_arms.csv).
    if (geom == GeomKind::Isoparam) {
#pragma omp parallel
        apply_residual_packed_isoparam_range(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
                d.nelements, d.p.data(), d.points, d.ux.data(), d.uy.data(), d.uz.data(), p.elems, p.ghost_buf.data(), p.ghost_idx, p.ghost_ptr, p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.n_ghost_entries, p.owned_nodes_ptr, rho, mu, rx, ry, rz, rc, scratch_n);
    } else {
#pragma omp parallel
        apply_residual_packed_affine_range(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
                d.adj_ptr, d.det_ptr, d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.rhie_chow_scale, d.ux.data(), d.uy.data(), d.uz.data(), p.elems, p.ghost_buf.data(), p.ghost_idx, p.ghost_ptr, p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.n_ghost_entries, p.owned_nodes_ptr, rho, mu, rx, ry, rz, rc, scratch_n, with_rc);
    }

    scalar_t *const fields[CVFEM_HEX8_N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (int f = 0; f < CVFEM_HEX8_N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + f * p.n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            fields[f][dest] += sum;
        }
    }
}

template <bool ISO>
static SFEM_NOINLINE void assemble_jacobian_packed(MeshData        &d,
                                                   PackedData      &p,
                                                   BSR4            &b,
                                                   const scalar_t   rho,
                                                   const scalar_t   mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p.max_actual_nodes_per_pack);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);
    // This sweep is scalar per element, not SIMD over a pack, so Rhie-Chow enters through
    // the same Hex8RhieChow the atomic assembly builds -- the only difference is that the
    // coordinates and the nodal gradient are read out of the pack rather than out of the
    // mesh. Both hand-written kernels below take the term; the generated ones and the
    // finite-difference reference do not, which the driver refuses rather than measures.
    const int    with_rc = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);


#pragma omp parallel
    // Six variants, chosen here instead of tested per pack. The default carries the remaining
    // values to the body's own final branch, which treated them alike already, so this
    // instantiates seven bodies rather than the enum's thirteen.
    // ONE ARM, as on the residual above. The four generated CSE arrangements and the
    // finite-difference Jacobian this switch could select are retired: Grace measures the
    // sum-factored assembly fastest everywhere it is the coloured or packed layout's kernel, and
    // the generated arrangement's only win -- the atomic layout -- ties packed sumfact at 14
    // MELEM/s and is half the coloured rate.
            assemble_jacobian_packed_range<ISO>(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d.adj_ptr, d.det_ptr, d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.rhie_chow_scale, d.ux.data(), d.uy.data(), d.uz.data(), p.elems, p.ghost_idx, p.ghost_mat_ptr.data(), p.ghost_mat_val.data(), p.ghost_ptr, p.local_element_slot.data(), p.local_global_slot_ptr.data(), p.local_rowptr_ptr.data(), p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.owned_nodes_ptr, b.values->data(), rho, mu, u_n, bsr_n, with_rc,
            cvfem_hex8_rc_config_for(d));

    CVFEM_PHASE_CLOCK(_tg);
    scalar_t *const SFEM_RESTRICT gvalues = b.values->data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const ptrdiff_t begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t end   = p.ghost_reduce_ptr[row + 1];
        for (ptrdiff_t j = begin; j < end; ++j) {
            const ptrdiff_t ghost_entry = p.ghost_reduce_idx[j];
            const ptrdiff_t k0          = p.ghost_mat_ptr[(size_t)ghost_entry];
            const ptrdiff_t k1          = p.ghost_mat_ptr[(size_t)ghost_entry + 1];
            for (ptrdiff_t t = k0; t < k1; ++t) {
                bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16], p.ghost_mat_val.data() + t * 16);
            }
        }
    }
    CVFEM_PHASE_GLOBAL(_tg, PH_GHOST);
}

template <bool ISO>
static SFEM_NOINLINE void apply_jacobian_action_packed(MeshData              &d,
                                                       PackedData            &p,
                                                       const scalar_t         rho,
                                                       const scalar_t         mu,
                                                       const scalar_t *const  dir,
                                                       scalar_t *const        jv,
                                                       const scalar_t *const SFEM_RESTRICT ugrad = nullptr,
                                                       const scalar_t *const SFEM_RESTRICT vgrad = nullptr,
                                                       const int              limiter = 0,
                                                       const scalar_t         venkat_c = scalar_t(0)) {
    const bool with_ho = ugrad != nullptr && vgrad != nullptr;
    // Hoisted out of the face loops -- see Hex8RhieChowPack::coeff. Its own cache key makes
    // this free after the first call, so --warmup absorbs the build and the timed loop
    // measures what the solver's Krylov iterations measure.
    cvfem_hex8_build_rc_coeff(d, rho, mu);
    const size_t scratch_n = packed_scratch_n(p.max_actual_nodes_per_pack);
    // Rhie-Chow staged per pack, exactly as apply_residual_packed does it and exactly as
    // the solver's own packed Jacobian does (cvfem_hex8_ns_packed.hpp). Slot 3 grows from
    // three arrays to six when it is on, and slot 4 carries the direction's reconstructed
    // gradient -- the term that makes this the *exact* Rhie-Chow Jacobian rather than the
    // frozen-gradient one the assembled matrix keeps.
    const int    with_rc   = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);
    const bool   with_qg   = with_rc && !d.qgx.empty();
    const size_t slot3_n   = with_rc ? packed_rc_n(p.max_actual_nodes_per_pack) : packed_xyz_n(p.max_actual_nodes_per_pack);


#pragma omp parallel
    apply_jacobian_action_packed_range<ISO>(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d.elems, d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc_coeff.data(), d.rc_w.data(), d.rhie_chow_scale, d.ux.data(), d.uy.data(), d.uz.data(), p.elems, p.ghost_buf.data(), p.ghost_idx, p.ghost_ptr, p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.n_ghost_entries, p.owned_nodes_ptr, rho, mu, dir, jv, ugrad, vgrad, limiter, venkat_c, with_ho, scratch_n, with_rc, with_qg, slot3_n,
            cvfem_hex8_rc_config_for(d),
            d.adj_ptr, d.det_ptr);

    CVFEM_PHASE_CLOCK(_tg);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        scalar_t *const    out   = jv + (ptrdiff_t)dest * CVFEM_HEX8_N_FIELDS;
        for (int f = 0; f < CVFEM_HEX8_N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + f * p.n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            out[f] += sum;
        }
    }
    CVFEM_PHASE_GLOBAL(_tg, PH_GHOST);
}

static SFEM_NOINLINE void apply_jacobian_action_packed_pa(MeshData             &d,
                                                          PackedData           &p,
                                                          const scalar_t        rho,
                                                          const scalar_t        mu,
                                                          const scalar_t *const dir,
                                                          scalar_t *const       jv) {
    const size_t scratch_n = packed_scratch_n(p.max_actual_nodes_per_pack);
    const int    with_rc   = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);
    const bool   with_qg   = with_rc && !d.qgx.empty();


#pragma omp parallel
    apply_jacobian_action_packed_pa_range(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d.adj_ptr, d.det_ptr, d.nelements, d.pa_tangent.data(), d.points, d.qgx.data(), d.qgy.data(), d.qgz.data(), d.rc_coeff.data(), d.rc_w.data(), d.rhie_chow_scale, p.elems, p.ghost_buf.data(), p.ghost_idx, p.ghost_ptr, p.max_actual_nodes_per_pack, p.n_elements_per_pack, p.n_ghost_entries, p.owned_nodes_ptr, rho, mu, dir, jv, scratch_n, with_rc, with_qg,
            cvfem_hex8_rc_config_for(d));

    CVFEM_PHASE_CLOCK(_tg);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        scalar_t *const    out   = jv + (ptrdiff_t)dest * CVFEM_HEX8_N_FIELDS;
        for (int f = 0; f < CVFEM_HEX8_N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + f * p.n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            out[f] += sum;
        }
    }
    CVFEM_PHASE_GLOBAL(_tg, PH_GHOST);
}

#endif  // CVFEM_HEX8_PACKED_LAUNCH_HPP
