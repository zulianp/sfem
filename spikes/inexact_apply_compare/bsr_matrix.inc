// A block CRS matrix for the benchmarks, so the assembled operator sits in the
// same table as the matrix-free and partially-assembled ones.
//
// BSR is the format SFEM uses for a vector-valued problem: the graph is the
// mesh's node-to-node graph and each entry is a `dim x dim` dense block, so a
// three-component displacement stores one ninth as many column indices as the
// scalar-expanded equivalent and its apply moves that much less index traffic.
//
// The graph is built here rather than borrowed from the library for the reason
// the mesh is: this harness links generated kernels and nothing else, and the
// whole point of comparing against BSR is to compare the *kernels*, not two
// different surrounding frameworks.
#ifndef SFEM_SPIKE_BSR_MATRIX_INC
#define SFEM_SPIKE_BSR_MATRIX_INC

#include <algorithm>
#include <vector>

struct BsrGraph {
    ptrdiff_t n_rows = 0;          //: nodes, not degrees of freedom
    std::vector<count_t> rowptr;
    std::vector<idx_t> colidx;
    ptrdiff_t nnz() const { return rowptr.empty() ? 0 : (ptrdiff_t)rowptr.back(); }
};

// Node-to-node adjacency: two nodes are neighbours when an element holds both.
// The diagonal is included, and each row is sorted, because the generated
// scatter finds a column by counting how many entries precede it.
static BsrGraph build_bsr_graph(const Mesh &m) {
    const int nxe = (int)m.ev.size();
    std::vector<std::vector<idx_t>> rows((size_t)m.nnodes);
    for (ptrdiff_t e = 0; e < m.nelements; ++e) {
        for (int i = 0; i < nxe; ++i) {
            const idx_t row = m.ev[i][e];
            for (int j = 0; j < nxe; ++j) {
                rows[(size_t)row].push_back(m.ev[j][e]);
            }
        }
    }
    BsrGraph g;
    g.n_rows = m.nnodes;
    g.rowptr.resize((size_t)m.nnodes + 1, 0);
    for (ptrdiff_t r = 0; r < m.nnodes; ++r) {
        auto &row = rows[(size_t)r];
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        g.rowptr[(size_t)r + 1] = g.rowptr[(size_t)r] + (count_t)row.size();
    }
    g.colidx.resize((size_t)g.nnz());
    for (ptrdiff_t r = 0; r < m.nnodes; ++r) {
        std::copy(rows[(size_t)r].begin(), rows[(size_t)r].end(),
                  g.colidx.begin() + g.rowptr[(size_t)r]);
    }
    return g;
}

// y = A x, blocks stored row-major inside the entry, components interleaved in
// the vectors.  One row per thread and no atomics: a row of a BSR matrix is
// written by exactly one thread, which is the whole reason an assembled apply
// scales differently from a matrix-free one.
template <int BS>
static void bsr_apply(const BsrGraph &g,
                      const double *const RSTR values,
                      const double *const RSTR x,
                      double *const RSTR y) {
#pragma omp parallel for
    for (ptrdiff_t row = 0; row < g.n_rows; ++row) {
        double acc[BS];
        for (int i = 0; i < BS; ++i) acc[i] = 0;
        const count_t begin = g.rowptr[(size_t)row];
        const count_t end = g.rowptr[(size_t)row + 1];
        for (count_t k = begin; k < end; ++k) {
            const double *const block = &values[(size_t)k * BS * BS];
            const double *const xv = &x[(size_t)g.colidx[(size_t)k] * BS];
            for (int i = 0; i < BS; ++i) {
                for (int j = 0; j < BS; ++j) acc[i] += block[i * BS + j] * xv[j];
            }
        }
        for (int i = 0; i < BS; ++i) y[(size_t)row * BS + i] = acc[i];
    }
}

#endif
