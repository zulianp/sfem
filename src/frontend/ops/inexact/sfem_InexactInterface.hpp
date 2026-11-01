#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

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
        class BsrStore {
        public:
            virtual ~BsrStore() = default;
            virtual int         update(const AssembleBSR &assemble)                    = 0;
            virtual void        apply(const real_t *const x, real_t *const y) const    = 0;  // y += A x
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

        /// diag += the stored operator's diagonal, interleaved with `stride` components.
        virtual void diagonal(const ptrdiff_t           nelements,
                              const idx_t *const *const elements,
                              const ptrdiff_t           stride,
                              real_t *const             diag) const = 0;

        virtual std::size_t bytes() const    = 0;
        virtual std::string describe() const = 0;
    };

}  // namespace sfem
