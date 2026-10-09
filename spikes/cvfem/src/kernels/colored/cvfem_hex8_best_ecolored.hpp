#ifndef CVFEM_HEX8_BEST_ECOLORED_HPP
#define CVFEM_HEX8_BEST_ECOLORED_HPP

// Element-coloured layout: the flat sweep of cvfem_hex8_best_atomic.hpp with the atomics
// removed. Elements are coloured so that no two of a colour share a node, the sweep runs one
// colour at a time, and every write into the global residual / action is a plain `+=`.
//
// This is the colouring the literature means -- Reguly and Giles, deal.II's matrix-free path,
// the GPU assembly work -- and it is NOT the pack colouring of cvfem_hex8_best_colored.hpp,
// which colours packs and keeps the packed layout's staging. Pack colouring is enough on a CPU
// where a pack is one thread; element colouring is what a comparison against the published
// baseline needs, and it is the arm that says how much of the packed format's margin is the
// layout rather than the absence of atomics.
//
// The element numbering is permuted into colour order at setup, by smesh::ElementColoring with
// `modify_mesh`, so a colour is a CONTIGUOUS element range and this file is
// the atomic sweep verbatim apart from the loop bounds and the write-back. That is deliberate:
// the 16-wide lane blocking, the memcpy geometry gather and the generated micro-kernels are
// identical, so what the measurement separates is the scatter strategy and not the kernel.
//
// Two consequences worth knowing. The sweep carries the higher-order deferred correction,
// which the pack-coloured sweep does not -- it takes no ugrad argument, which is why
// jobs/dram_traffic.sbatch skips it for the -ho arms. And each node takes exactly one
// contribution per colour, in colour order, so the summation order is fixed by the colouring:
// reproducible as long as the colouring is, which it is for a fixed mesh and element order.

#include "kernels/microkernels/hex8/affine/cvfem_hex8_ns_upwind_affine.hpp"
#include "kernels/cvfem_range.hpp"
#include "kernels/standard/cvfem_hex8_best_atomic.hpp"


// BOTH SWEEPS ARE AFFINE, and they are in affine/ so that the assumption is in the path rather
// than only in a comment: each reads one adjugate and one determinant per element out of the
// precomputed table. There is no isoparametric element-coloured sweep -- see
// isoparametric/README.md for what it would take and why it has not been worth building.
#include "kernels/colored/affine/cvfem_hex8_best_ecolored_affine.hpp"

#endif  // CVFEM_HEX8_BEST_ECOLORED_HPP
