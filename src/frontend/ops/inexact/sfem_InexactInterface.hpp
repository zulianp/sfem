#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "sfem_Team.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "smesh_types.hpp"

// What a generated operator sees of its stored tangent, wherever the stores run: the set-up's
// input, the packed traversal's view and the store interfaces.  The host stores
// (sfem_InexactStore.hpp) and the device ones (cuda/sfem_cuda_InexactStore.cuh) implement them.

namespace sfem {
    namespace inexact {

        /// What the set-up reads: the state, the geometry and the material.  The geometry is the
        /// element's affine adjugate and determinant (SoA, one per element) when the operator
        /// caches them, or the node coordinates otherwise, from which the set-up forms the
        /// Jacobian at every point -- the stored tangent is pulled back per point, so a curved
        /// element is stored exactly as the exact operator integrates it.
        ///
        /// A device store reads what a device operator hands its kernels: `elements` and `points`
        /// are device arrays of device pointers, `adjugate` and `quadrature` host arrays of device
        /// pointers, the vectors device vectors; `parameters` is on the host.
        struct SetupInput {
            ptrdiff_t                nelements{0};
            const idx_t *const      *elements{nullptr};     // [node][element], mesh order
            const geom_t *const     *adjugate{nullptr};     // [9][element], affine
            const geom_t            *determinant{nullptr};  // [element], affine
            const geom_t *const     *points{nullptr};       // [3][node], isoparametric
            const real_t            *parameters{nullptr};   // the unit's material constants, in its order
            ptrdiff_t                u_stride{3};
            const real_t            *u{nullptr};            // current state, interleaved
            const real_t            *z{nullptr};            // previous state, when the unit reads one
            // The integration-point fields the unit reads (a relaxation group's history), one base
            // pointer each, laid out [component][point][element] at `quadrature_stride` per point.
            const real_t *const     *quadrature{nullptr};
            ptrdiff_t                quadrature_stride{0};
            // Every point's tangent projected to positive semidefinite (Op::set_tangent_projection).
            bool                     project{false};
        };

        /// One block of the packed mesh, as the two-pass traversal reads it (`smesh::PackedMesh`).
        struct PackedView {
            ptrdiff_t                 n_packs{0};
            ptrdiff_t                 n_elements_per_pack{0};
            ptrdiff_t                 max_nodes_per_pack{0};
            const uint16_t *const    *elements{nullptr};  // [node][element], pack-local
            const ptrdiff_t          *owned_nodes_ptr{nullptr};
            ptrdiff_t                 n_ghost_entries{0};
            const ptrdiff_t          *ghost_ptr{nullptr};
            const idx_t              *ghost_idx{nullptr};
            ptrdiff_t                 n_ghost_reduce_rows{0};
            const ptrdiff_t          *ghost_reduce_ptr{nullptr};
            const ptrdiff_t          *ghost_reduce_idx{nullptr};
            const idx_t              *ghost_reduce_dest{nullptr};
            real_t                   *ghost_buf{nullptr};  // 3 * n_ghost_entries
            ptrdiff_t                 n_nodes{0};          // the mesh's, which the outputs span
        };

        /// What a BSR store asks of its operator: the operator's tangent assembled into the given
        /// node graph, in the given value type (`Op::hessian_bsr_typed`).
        using AssembleBSR = std::function<int(const count_t *rowptr, const idx_t *colidx, smesh::PrimitiveType type, void *values)>;

        /// The quantized assembled operator (`store: bsr`): the operator's own tangent, assembled in
        /// single precision on the space's node graph and encoded in place (sfem_QuantBSR.hpp).
        ///
        /// The difference form is exact for an operator that annihilates translations.  One that
        /// does not -- a time scheme's inertia, whose mass matrix rows sum to the lumped mass --
        /// keeps the rows whose blocks do not sum to zero as a correction, `y_i += R_i x_i` with
        /// `R_i = sum_j K_ij` in single precision: exact for any operator, and empty in a
        /// quasi-static solve, where the sums are the assembly's rounding.
        /// What a contact step adds to the stored tangent's diagonal blocks, applied in the store's
        /// own pass: the obstacle's curvature, a symmetric 3x3 block per contact node (packed
        /// [xx, xy, xz, yy, yz, zz], node_obstacle_pack_sym), and a shift times |reference| per
        /// dof. Pointers where the store's vectors live.
        struct DiagonalExtra {
            const idx_t   *contact_of{nullptr};  // per node: its contact's index, -1 for none; null: no curvature
            const real_t  *blocks{nullptr};      // 6 per contact
            real_t         shift{0};
            const real_t  *reference{nullptr};   // per dof, read where shift != 0
        };

        /// out += B in, B packed as DiagonalExtra's blocks (node_obstacle_packed_sym_apply's arithmetic).
        SFEM_HOST_DEVICE SFEM_INLINE void packed_sym_apply(const real_t *const B, const real_t *const in, real_t *const out) {
            out[0] += B[0] * in[0] + B[1] * in[1] + B[2] * in[2];
            out[1] += B[1] * in[0] + B[3] * in[1] + B[4] * in[2];
            out[2] += B[2] * in[0] + B[4] * in[1] + B[5] * in[2];
        }

        class BsrStore {
        public:
            virtual ~BsrStore() = default;
            virtual int         update(const AssembleBSR &assemble)                    = 0;
            virtual void        apply(const real_t *const x, real_t *const y) const    = 0;  // y += A x
            /// y = A x + E x (`overwrite`) or y += A x + E x, E the extra diagonal blocks, in one pass
            /// where the store can: the operations of apply, then the curvature, then the shift.
            virtual void apply(const real_t *const x, real_t *const y, const DiagonalExtra &extra, const bool overwrite) const = 0;
            /// y += A x by the threads of a region (sfem_TeamOperator.hpp): on return the calling
            /// thread's rows, team's share of the 3 n entries, are complete. The host's.
            virtual void team_apply(const Team &team, const real_t *const x, real_t *const y) const {
                (void)team, (void)x, (void)y;
                SFEM_ERROR("inexact: this bsr store has no team apply\n");
            }
            virtual void        add_diagonal(real_t *const diag) const                 = 0;
            virtual std::size_t bytes() const                                          = 0;
            virtual std::string describe() const                                       = 0;
        };

    }  // namespace inexact

    /// One domain's stored tangent, whatever its encoding: what a generated operator holds
    /// between `inexact_update` and the applies of a Newton step.
    class InexactStore {
    public:
        virtual ~InexactStore() = default;

        /// Evaluates the tangent at every point from the state and encodes it.
        virtual void update(const inexact::SetupInput &input) = 0;

        /// `out += A h`, interleaved vectors of `stride` components.  Packed when `packed` is
        /// given and the options allow it, atomic otherwise.
        virtual void apply(const ptrdiff_t                  nelements,
                           const idx_t *const *const        elements,
                           const inexact::PackedView *const packed,
                           const ptrdiff_t                  stride,
                           const real_t *const              h,
                           real_t *const                    out) const = 0;

        /// The same, by the threads of a region, over the elements [element_begin, element_end)
        /// (a distributed product's interior or interface, sfem_ElementScope.hpp): on return the
        /// calling thread's share of the rows, team's of the space's `stride` n entries, is
        /// complete. The packed traversal walks the whole block, so a part of it goes atomic.
        /// The host's.
        virtual void team_apply(const Team                      &team,
                                const ptrdiff_t                  nelements,
                                const ptrdiff_t                  element_begin,
                                const ptrdiff_t                  element_end,
                                const idx_t *const *const        elements,
                                const inexact::PackedView *const packed,
                                const ptrdiff_t                  stride,
                                const real_t *const              h,
                                real_t *const                    out) const {
            (void)team, (void)nelements, (void)element_begin, (void)element_end, (void)elements, (void)packed, (void)stride,
                    (void)h, (void)out;
            SFEM_ERROR("inexact: this store has no team apply\n");
        }

        /// diag += the stored operator's diagonal, interleaved with `stride` components.
        virtual void diagonal(const ptrdiff_t           nelements,
                              const idx_t *const *const elements,
                              const ptrdiff_t           stride,
                              real_t *const             diag) const = 0;

        virtual std::size_t bytes() const    = 0;
        virtual std::string describe() const = 0;
    };

}  // namespace sfem
