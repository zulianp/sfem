#pragma once

// The semi-structured prolongation as a sparse matrix, derived rather than probed.
//
// The driver needs Galerkin coarse operators, and it cannot compose them matrix-free without
// a coarse level reaching back into a finer one. Until now it recovered them by *probing*:
// colouring the coarse graph and applying the operator to unit vectors, once per colour per
// component. That is numerically rediscovering something the lattice already determines, and
// it has a failure mode probing cannot avoid -- the sparsity pattern has to be guessed in
// advance, and an entry falling outside the guess is folded into the wrong slot rather than
// dropped, so too narrow a pattern yields a wrong matrix. The retry loop that widened the
// guess is what produced a dense 7,144,929-block coarse operator and 10,692 operator
// applications per Newton step on a 2,673-node macro mesh.
//
// Everything about the transfer is a closed-form function of lattice position. With
// q = to_level / from_level, a fine node at (xi, yi, zi) decomposes as xi = ax*q + rx, and
// interpolates from the coarse cell corner (ax, ay, az): two coarse nodes per axis when the
// offset is non-zero, one when it is aligned.
//
//     aligned in x, y and z   ->  1 entry
//     one axis off-lattice    ->  2 entries
//     two axes                ->  4 entries
//     three axes              ->  8 entries
//
// For the 2:1 hops that make up the standard hierarchy (16 -> 8 -> 4 -> 2 -> 1) the weights
// are products of halves, so **every weight in a row is 1/nnz for that row** -- the values
// carry no information the row length does not already carry, and computing one is a
// reciprocal of a small integer rather than three multiplications. A general q needs the
// trilinear product, but even then the weight depends only on (rx, ry, rz) and never on the
// element or the node id, so it is a per-offset table and never a per-entry array.
//
// The values array below therefore exists only because `sfem::rap` takes a CRS and reads
// one. `apply_structured` is the same operator with no values at all, and is what a
// structure-aware triple product would use; if that is ever written, the array goes away
// entirely.

#include "sfem_API.hpp"
#include "sfem_CRS.hpp"

#include "smesh_sshex8.hpp"

#include <memory>
#include <vector>

namespace cvfem_ss {

    struct ProlongationPattern {
        ptrdiff_t                    n_fine{0};
        ptrdiff_t                    n_coarse{0};
        int                          from_level{0};
        int                          to_level{0};
        std::vector<sfem::count_t>   rowptr;
        std::vector<sfem::idx_t>     colidx;
        // True when every row's weights are 1/nnz, i.e. the ratio is 2:1. Then the values
        // are implied by rowptr alone and nothing needs storing.
        bool                         uniform{false};
    };

    // Symbolic pass. Row lengths follow from the offsets, so this needs no arithmetic on
    // values and no knowledge of the operator.
    inline void build_prolongation_pattern(const ptrdiff_t                  nelements,
                                           const int                        from_level,
                                           const smesh::idx_t *const *const from_elements,
                                           const int                        to_level,
                                           const smesh::idx_t *const *const to_elements,
                                           const ptrdiff_t                  n_fine,
                                           const ptrdiff_t                  n_coarse,
                                           ProlongationPattern             &out) {
        const int q = to_level / from_level;

        out.n_fine     = n_fine;
        out.n_coarse   = n_coarse;
        out.from_level = from_level;
        out.to_level   = to_level;
        out.uniform    = (q == 2);

        out.rowptr.assign((size_t)n_fine + 1, 0);

        // Row lengths. A node shared between macro-elements is visited more than once and
        // every visit writes the same length, so assignment is idempotent and the repeated
        // work is harmless.
        for (int zi = 0; zi <= to_level; ++zi) {
            for (int yi = 0; yi <= to_level; ++yi) {
                for (int xi = 0; xi <= to_level; ++xi) {
                    const int nx  = (xi % q) ? 2 : 1;
                    const int ny  = (yi % q) ? 2 : 1;
                    const int nz  = (zi % q) ? 2 : 1;
                    const int nnz = nx * ny * nz;
                    const int l   = smesh::sshex8_lidx(to_level, xi, yi, zi);
                    for (ptrdiff_t e = 0; e < nelements; ++e)
                        out.rowptr[(size_t)to_elements[l][e] + 1] = (sfem::count_t)nnz;
                }
            }
        }
        for (ptrdiff_t i = 0; i < n_fine; ++i) out.rowptr[(size_t)i + 1] += out.rowptr[(size_t)i];

        out.colidx.assign((size_t)out.rowptr[(size_t)n_fine], 0);

        for (int zi = 0; zi <= to_level; ++zi) {
            for (int yi = 0; yi <= to_level; ++yi) {
                for (int xi = 0; xi <= to_level; ++xi) {
                    const int ax = xi / q, rx = xi % q;
                    const int ay = yi / q, ry = yi % q;
                    const int az = zi / q, rz = zi % q;
                    const int nx = rx ? 2 : 1, ny = ry ? 2 : 1, nz = rz ? 2 : 1;
                    const int l  = smesh::sshex8_lidx(to_level, xi, yi, zi);

                    for (ptrdiff_t e = 0; e < nelements; ++e) {
                        sfem::count_t at = out.rowptr[(size_t)to_elements[l][e]];
                        for (int k = 0; k < nz; ++k)
                            for (int j = 0; j < ny; ++j)
                                for (int i = 0; i < nx; ++i) {
                                    const int cl = smesh::sshex8_lidx(from_level, ax + i, ay + j, az + k);
                                    out.colidx[(size_t)at++] = (sfem::idx_t)from_elements[cl][e];
                                }
                    }
                }
            }
        }
    }

    // Build from a coarse/fine FunctionSpace pair, mirroring what
    // create_hierarchical_prolongation does to reach the lattice: the coarse side is a
    // semi-structured space at its own level, or -- for the last hop, where derefine yields
    // an unstructured HEX8 space -- level 1 via hex8_elements_as_sshex8_level1.
    inline void build_from_spaces(const std::shared_ptr<sfem::FunctionSpace> &from_space,  // coarse
                                  const std::shared_ptr<sfem::FunctionSpace> &to_space,    // fine
                                  ProlongationPattern                        &out) {
        auto &to_m   = to_space->mesh();
        auto &from_m = from_space->mesh();

        if (!to_space->has_semi_structured_mesh())
            SFEM_ERROR("build_from_spaces: the fine space must be semi-structured\n");
        if (to_m.n_blocks() != 1 || from_m.n_blocks() != 1)
            SFEM_ERROR("build_from_spaces: multi-block is not implemented\n");

        auto      to_b     = to_m.block(0);
        auto      from_b   = from_m.block(0);
        const int to_level = smesh::semistructured_level(to_m);

        const int    bs       = to_space->block_size();
        const ptrdiff_t n_fine   = to_space->n_dofs() / bs;
        const ptrdiff_t n_coarse = from_space->n_dofs() / from_space->block_size();

        if (smesh::Env::read<int>("SFEM_SS_TRANSFER_DEBUG", 0)) {
            std::printf("    [xfer] from_ss=%d to_level=%d ne_to=%td ne_from=%td n_fine=%td n_coarse=%td\n",
                        (int)from_space->has_semi_structured_mesh(), to_level,
                        (ptrdiff_t)to_b->n_elements(), (ptrdiff_t)from_b->n_elements(), n_fine, n_coarse);
        }

        if (from_space->has_semi_structured_mesh()) {
            build_prolongation_pattern(to_b->n_elements(),
                                       smesh::semistructured_level(from_m),
                                       from_b->elements()->data(),
                                       to_level,
                                       to_b->elements()->data(),
                                       n_fine, n_coarse, out);
        } else {
            // Last hop: the coarse space is unstructured HEX8, and the reference kernel
            // (sshex8_hierarchical_prolongation) reads the coarse values through the *SS
            // mesh's own macro-corner slots* rather than through the unstructured mesh's
            // element array. Take the columns from the same place; that needs no assumption
            // about how the derefined mesh numbers its nodes, and going via
            // hex8_elements_as_sshex8_level1 on the coarse array instead gives a wrong
            // matrix (measured: relative error 1.14 against the matrix-free transfer).
            smesh::idx_t *from_corners[8];
            for (int k = 0; k < 2; ++k)
                for (int j = 0; j < 2; ++j)
                    for (int i = 0; i < 2; ++i)
                        from_corners[smesh::sshex8_lidx(1, i, j, k)] =
                                to_b->elements()->data()[smesh::sshex8_lidx(
                                        to_level, i * to_level, j * to_level, k * to_level)];

            build_prolongation_pattern(to_b->n_elements(),
                                       1,
                                       from_corners,
                                       to_level,
                                       to_b->elements()->data(),
                                       n_fine, n_coarse, out);
        }
    }

    // Apply without any values array: the weight is 1/nnz for a 2:1 hop, and otherwise comes
    // from the offsets. This is the form the philosophy wants; it is used by the gate, and
    // would be used by a structure-aware triple product.
    inline void apply_structured(const ProlongationPattern &p, const real_t *const coarse, real_t *const fine) {
        if (!p.uniform) SFEM_ERROR("apply_structured: only the 2:1 hop is implied by row length\n");
        for (ptrdiff_t i = 0; i < p.n_fine; ++i) {
            const sfem::count_t b = p.rowptr[(size_t)i], e = p.rowptr[(size_t)i + 1];
            const real_t        w = real_t(1) / (real_t)(e - b);
            real_t              s = 0;
            for (sfem::count_t k = b; k < e; ++k) s += coarse[p.colidx[(size_t)k]];
            fine[i] += w * s;
        }
    }

    // Materialise as a CRS, because sfem::rap takes one and reads its values. For the 2:1
    // hop this writes 1/nnz and does no trilinear arithmetic at all.
    inline std::shared_ptr<sfem::CRS<sfem::count_t, sfem::idx_t, real_t, real_t>>
    to_crs(const ProlongationPattern &p) {
        const ptrdiff_t nnz = (ptrdiff_t)p.rowptr[(size_t)p.n_fine];

        auto rowptr = smesh::create_host_buffer<sfem::count_t>((size_t)p.n_fine + 1);
        auto colidx = smesh::create_host_buffer<sfem::idx_t>((size_t)nnz);
        auto values = smesh::create_host_buffer<real_t>((size_t)nnz);

        std::copy(p.rowptr.begin(), p.rowptr.end(), rowptr->data());
        std::copy(p.colidx.begin(), p.colidx.end(), colidx->data());

        const int q = p.to_level / p.from_level;
        if (p.uniform) {
            for (ptrdiff_t i = 0; i < p.n_fine; ++i) {
                const sfem::count_t b = p.rowptr[(size_t)i], e = p.rowptr[(size_t)i + 1];
                const real_t        w = real_t(1) / (real_t)(e - b);
                for (sfem::count_t k = b; k < e; ++k) values->data()[(size_t)k] = w;
            }
        } else {
            // A non-2:1 hop (an L=24 hierarchy has a 3:1 one) has non-uniform weights. They
            // are still a function of the offsets alone, so the fix is to carry the offset
            // per row rather than to store a weight per entry. Not needed by any hierarchy
            // this driver builds, so it errors rather than guessing.
            SFEM_ERROR("to_crs: only 2:1 hops are implemented (got %d:1)\n", q);
        }

        return sfem::h_crs_spmv<sfem::count_t, sfem::idx_t, real_t, real_t>(
                p.n_fine, p.n_coarse, rowptr, colidx, values, real_t(0));
    }

}  // namespace cvfem_ss
