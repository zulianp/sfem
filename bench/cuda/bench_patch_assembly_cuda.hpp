#pragma once

#include <cstddef>

#include "sfem_base.hpp"

// The device assembly strategies of the Mooney-Rivlin Prony TET10 tangent into the node graph's BSR,
// compared on one mesh with the same inputs (bench_patch_assembly.cpp reads the mesh, the .cu runs):
//   coop     the library's cooperative kernel (cu_mooney_rivlin_prony_tet10_hessian_bsr_i_msoa): a thread
//            per element and point, node-pair blocks added with atomics;
//   patch    a thread per row: the elements of its patch in a fixed order, each element's points'
//            tangents recomputed for the row, its blocks summed in double and added to the row it alone
//            writes (no atomics, the same bits every run);
// each into f64 and f32 values, all rows at once and by windows of rows (the stored tangent's update).
struct PatchBenchInput {
    ptrdiff_t                    nelements{0}, nnodes{0};
    const idx_t *const          *elements{nullptr};  // host [10][nelements]
    const geom_t *const         *points{nullptr};    // host [3][nnodes]
    const count_t               *rowptr{nullptr};    // host node graph
    const idx_t                 *colidx{nullptr};
    const count_t               *n2e_ptr{nullptr};   // host node -> elements, ascending
    const element_idx_t         *n2e_idx{nullptr};
    int                          windows{4};
    int                          repeats{5};
    bool                         patch{true};  // the thread-per-row kernel (slow: SFEM_BENCH_PATCH=0 skips it)
};

int run_patch_bench(const PatchBenchInput &in);
