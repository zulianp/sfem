#pragma once

// Element-wise Galerkin coarsening, as seen by a driver.
//
// The kernel lives in cvfem_ss_galerkin.hpp and is compiled into its own translation unit,
// for the same reason cvfem_hex8_ns_op.cpp exists: the CVFEM element headers define
// file-scope names (N_FIELDS, scalar_t, usage) that collide with a driver's own, so the
// operator keeps them to itself and exposes an sfem-typed surface. This header is that
// surface for the Galerkin path -- sfem and standard types only, no CVFEM kernel headers.

#include "cvfem_hex8_ns_op.hpp"

#include "sfem_BSR.hpp"
#include "sfem_Function.hpp"

#include <memory>
#include <vector>

namespace cvfem_ss {

    using CoarseBSR = sfem::BSR<sfem::count_t, sfem::idx_t, real_t, real_t>;

    // Assemble the Galerkin coarse operator P^T A P for `coarse` directly from the fine
    // operator's macro-elements, with no probing and no guessed sparsity pattern.
    //
    // `op` must be an initialised semi-structured operator whose update() has run for the
    // state being linearised about -- the assembly reads the same nodal fields and pressure
    // gradient the apply reads. `fine` is that operator's space. The ratio q is taken from
    // the two levels and must divide the fine one.
    //
    // The result is unconstrained: it is P^T A P for the operator as assembled, exactly as
    // the probe path's raw output was, so the caller applies the same constraint treatment
    // it already applies (column masking on the fine side, identity rows on the coarse).
    //
    // When `diag_out` is non-null it receives the coarse block diagonal, nnodes * 16
    // row-major 4x4 blocks, read off the assembled matrix.
    //
    // `fine_constrained`, when given, is one byte per fine dof (node * 4 + component) and
    // zeroes those columns, yielding P^T (A Z) P -- the composite the probe path recovered.
    // The caller still patches the coarse identity rows.
    std::shared_ptr<CoarseBSR> assemble_coarse_operator(const sfem::CVFEMNavierStokes              &op,
                                                        const std::shared_ptr<sfem::FunctionSpace> &coarse,
                                                        const std::shared_ptr<sfem::FunctionSpace> &fine,
                                                        std::vector<real_t> *const                  diag_out,
                                                        const uint8_t *const fine_constrained = nullptr);

    // Every coarse operator in the hierarchy, built element-wise.
    //
    // Level 1 comes from the micro-cell matrices; each level below is one element-local
    // coarsening hop from the level above. The hops reproduce the composite the driver's
    // transfers define -- Z_i Rhat A_{i-1} Z_{i-1} Phat -- by masking the source level's
    // columns with that level's own constraints, which SFEM already provides because
    // create_gmg_data derefines the Function at every level. Nothing global is built along
    // the way; only the returned matrices are assembled.
    //
    // `spaces[0]` is the fine space and `spaces[i]` level i; `masks[i]` is one byte per dof of
    // level i (node * 4 + component). Entry 0 of each returned vector is unused.
    //
    // With `element_matrices`, every level but the coarsest is kept as element matrices and
    // never assembled -- the hops coarsen element matrices to element matrices, so no global
    // sparse structure is built for them at all. The coarsest is still assembled, because it is
    // the one that gets factorised. Without it every level is assembled, which is the form the
    // probe path produced and the one the gates compare against.
    //
    // Assembled levels come back raw, so the caller applies the identity-row patch it already
    // applies; `op` and `diag` carry that treatment already for either form.
    struct CoarseHierarchy {
        // Assembled form, null on any level kept as element matrices.
        std::vector<std::shared_ptr<CoarseBSR>>              A;
        // Apply for every level, whichever form backs it. Constrained rows behave as identity,
        // so this is the operator the driver would have got from patch_identity_rows.
        std::vector<std::shared_ptr<sfem::Operator<real_t>>> op;
        // Block diagonal the smoothers invert, nnodes * 16 per level, identity on constrained
        // rows for the same reason.
        std::vector<std::vector<real_t>>                     diag;
    };

    // `state` is the fine-level state being linearised about. It is passed rather than assumed
    // because the operator's cached fields are whatever its last apply(), gradient() or
    // update() left there, which makes an assembly that just reads them depend on call order --
    // a dependency that is invisible when it holds and a wrong linearisation when it does not.
    CoarseHierarchy assemble_hierarchy(sfem::CVFEMNavierStokes                                 &op,
                                       const real_t *const                                      state,
                                       const std::vector<std::shared_ptr<sfem::FunctionSpace>> &spaces,
                                       const std::vector<std::vector<uint8_t>>                 &masks,
                                       const bool                                               element_matrices);

    // A coarse level kept as element matrices instead of assembled.
    //
    // The returned operator applies sum_e P_e^T A_e P_e directly from the macro-elements, so
    // this level never builds a sparse matrix; only the coarsest level, which is factorised,
    // still needs one. Constrained coarse rows behave as identity, matching what
    // patch_identity_rows gives the assembled form, and the apply accumulates into `y` like
    // every other operator here.
    //
    // Storage and flops are both larger than the assembled form by the duplication at shared
    // macro-element faces -- about 1.42x at a level-8 coarse lattice, 1.95x at level 4,
    // improving as the lattice deepens -- traded for contiguous blocks and no column
    // indirection. Which wins is a measurement.
    std::shared_ptr<sfem::Operator<real_t>> make_element_matrix_level(
            const sfem::CVFEMNavierStokes              &op,
            const std::shared_ptr<sfem::FunctionSpace> &coarse,
            const std::shared_ptr<sfem::FunctionSpace> &fine,
            std::vector<real_t> *const                  diag_out,
            const uint8_t *const                        fine_constrained,
            const uint8_t *const                        coarse_constrained);

    // The fine operator as per-macro-element lattice stencils, with no global matrix.
    //
    // The element-wise Galerkin assembly at q = 1 already produces exactly this: 27 fixed slots
    // per lattice node of each macro-element, slot s holding the block coupling a node to its
    // lattice neighbour at offset s. Accumulating those slots into a global BSR -- which is what
    // assemble_coarse_operator goes on to do -- exists only so the smoother can read them back
    // one patch at a time, through a binary search per block and a column-index indirection per
    // row. This returns the stencils themselves.
    //
    // One correction is needed to make a stencil hold the operator's own entries rather than its
    // macro-element's share of them. A node pair on a face, edge or corner shared between
    // macro-elements is assembled once per macro-element that contains both nodes, and the
    // operator's entry is the sum. The slots of such a pair are therefore folded: summed once and
    // written back to every macro-element that carries them, in a fixed order, so the result is
    // the same on any thread count. The fold map is derived from the cached pattern and reused
    // across Newton steps like the pattern itself.
    //
    // Storage is nmacro * (L+1)^3 * 27 blocks against n_nodes * 27 for the assembled form -- the
    // excess is the duplication at shared faces, about 1.37x on the FDA nozzle at level 8 -- and
    // it replaces both the double-precision BSR and the narrowed copy of its values that the
    // sweep reads today.
    //
    // `single` selects the storage precision, matching the smoother's own setting: the values are
    // in `vf` when true and in `vd` when false. `fine_constrained`, when given, zeroes the
    // columns of constrained dofs exactly as assemble_coarse_operator does.
    struct FineStencil {
        int                      L{0};       // lattice level of the macro-elements
        int                      nc{0};      // (L+1)^3, lattice nodes per macro-element
        ptrdiff_t                nmacro{0};
        std::vector<sfem::idx_t> gid;        // nmacro * nc, lattice node -> global node
        std::vector<float>       vf;         // nmacro * nc * 27 * 16 when single
        std::vector<real_t>      vd;         // the same, when not
    };

    std::shared_ptr<FineStencil> assemble_fine_stencil(const sfem::CVFEMNavierStokes              &op,
                                                       const std::shared_ptr<sfem::FunctionSpace> &space,
                                                       const uint8_t *const                        fine_constrained,
                                                       const bool                                  single);

    // Diagonal Vanka smoother, as an sfem operator applying M^-1.
    //
    // Replaces the nodal 4x4 block solve with a coupled solve over each micro-element patch
    // (8 corners, 32 dofs), with the velocity block approximated by its diagonal so the
    // velocities eliminate and an 8x8 pressure Schur complement per cell remains. That
    // recovers the velocity-pressure coupling a point-block smoother discards, which is why
    // block-Jacobi measures rho = 0.981 here.
    //
    // `state` is the linearisation point; the factorisations are valid for one Newton step and
    // the operator holds them, so it must be rebuilt when the state changes -- the same
    // lifetime as the element-wise Galerkin coarse operators. `constrained` is one byte per
    // dof; those dofs receive no correction.
    //
    // Additive over patches, averaged by patch multiplicity, accumulated through the two-pass
    // scatter, so a sweep is bitwise reproducible on any thread count.
    std::shared_ptr<sfem::Operator<real_t>> make_diagonal_vanka(
            sfem::CVFEMNavierStokes &op, const std::shared_ptr<sfem::FunctionSpace> &space,
            const real_t *const state, const uint8_t *const constrained, const real_t omega);

    // Same smoother for a coarse level, built from a matrix that already exists.
    //
    // A V-cycle is limited by its worst level, and the coarse levels were still running the
    // point-block smoother the fine level just replaced. They need no new assembly: the
    // element-wise Galerkin path has already produced each level's BSR, and each level's
    // operator carries its own lattice, which is all vanka_setup wants.
    //
    // `op` supplies the lattice only, so it must be the operator for THAT level; `A` is that
    // level's assembled matrix with its identity rows already patched.
    std::shared_ptr<sfem::Operator<real_t>> make_diagonal_vanka_from_bsr(
            sfem::CVFEMNavierStokes &op, const std::shared_ptr<CoarseBSR> &A,
            const uint8_t *const constrained, const real_t omega);

}  // namespace cvfem_ss
