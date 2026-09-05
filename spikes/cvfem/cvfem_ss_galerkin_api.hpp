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
#include "sfem_base.h"

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

}  // namespace cvfem_ss
