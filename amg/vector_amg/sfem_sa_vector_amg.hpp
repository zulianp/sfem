#ifndef SFEM_SA_VECTOR_AMG_HPP
#define SFEM_SA_VECTOR_AMG_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "sfem_BSR.hpp"
#include "sfem_CRS.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"

// TODO:
// Refactor the code using the new BSR API that supports rectangular block sizes
// Make sure to not use CRS when BSR is the optimal representation
// Seize opportunities to parallelize the code with OpenMP and make the code SIMD friendly

namespace sfem {

    template <typename C>
    struct SAVectorAMGAggregates {
        SharedBuffer<C> aggregate;
        ptrdiff_t       n_aggregates{0};
    };

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    struct SAVectorAMGLevel {
        SAVectorAMGAggregates<C>                aggregates;
        std::shared_ptr<BSR<R, C, TStorage, T>> a;
        std::shared_ptr<CRS<R, C, T, T>>        p;
        std::shared_ptr<CRS<R, C, T, T>>        r;
        std::shared_ptr<BSR<R, C, T, T>>        coarse_a;
        int                                     n_rigid_body_modes{0};
    };

    template <typename R, typename C, typename T>
    struct SAVectorAMGHierarchy {
        std::vector<SAVectorAMGLevel<R, C, T, T>> levels;
    };

    inline int sa_vector_amg_n_rigid_body_modes(const int block_size, const int spatial_dim) {
        const int dim = std::min(block_size, spatial_dim);
        if (dim <= 1) return 1;
        if (dim == 2) return 3;
        return 6;
    }

    template <typename X, typename T>
    inline T sa_vector_amg_rbm_value(const int                    block_size,
                                     const int                    spatial_dim,
                                     const int                    component,
                                     const int                    mode,
                                     const X* const* const        points,
                                     const ptrdiff_t              node,
                                     const T* const SFEM_RESTRICT centroid) {
        const int dim = std::min(block_size, spatial_dim);

        if (mode < dim) {
            return component == mode ? static_cast<T>(1) : static_cast<T>(0);
        }

        if (dim == 2) {
            const T x = static_cast<T>(points[0][node]) - centroid[0];
            const T y = static_cast<T>(points[1][node]) - centroid[1];
            return component == 0 ? -y : (component == 1 ? x : static_cast<T>(0));
        }

        const T x = static_cast<T>(points[0][node]) - centroid[0];
        const T y = static_cast<T>(points[1][node]) - centroid[1];
        const T z = static_cast<T>(points[2][node]) - centroid[2];

        switch (mode - dim) {
            case 0:
                return component == 1 ? -z : (component == 2 ? y : static_cast<T>(0));
            case 1:
                return component == 0 ? z : (component == 2 ? -x : static_cast<T>(0));
            default:
                return component == 0 ? -y : (component == 1 ? x : static_cast<T>(0));
        }
    }

    template <typename R, typename C>
    SAVectorAMGAggregates<C> sa_vector_amg_aggregate(const ptrdiff_t             block_rows,
                                                     const SharedBuffer<R>&      rowptr,
                                                     const SharedBuffer<C>&      colidx,
                                                     const SharedBuffer<mask_t>& boundary_nodes     = nullptr,
                                                     const int                   max_aggregate_size = 8) {
        auto     aggregates = create_host_buffer<C>(block_rows);
        C* const a          = aggregates->data();

        for (ptrdiff_t i = 0; i < block_rows; ++i) {
            a[i] = static_cast<C>(-1);
        }

        if (boundary_nodes) {
            const mask_t* const bdy = boundary_nodes->data();
            for (ptrdiff_t i = 0; i < block_rows; ++i) {
                if (bdy[i]) {
                    a[i] = static_cast<C>(-1);
                }
            }
        }

        ptrdiff_t           n_aggregates = 0;
        const R* const      rp           = rowptr->data();
        const C* const      ci           = colidx->data();
        const mask_t* const bdy          = boundary_nodes ? boundary_nodes->data() : nullptr;

        std::vector<C> frontier;
        frontier.reserve(std::max(1, max_aggregate_size));

        for (ptrdiff_t i = 0; i < block_rows; ++i) {
            if (a[i] >= 0) continue;
            if (bdy && bdy[i]) continue;

            const C agg = static_cast<C>(n_aggregates++);
            a[i]        = agg;

            frontier.clear();
            frontier.push_back(static_cast<C>(i));

            for (size_t head = 0; head < frontier.size() && static_cast<int>(frontier.size()) < max_aggregate_size; ++head) {
                const C current = frontier[head];
                for (R k = rp[current]; k < rp[current + 1] && static_cast<int>(frontier.size()) < max_aggregate_size; ++k) {
                    const C j = ci[k];
                    if (j < 0 || static_cast<ptrdiff_t>(j) >= block_rows || a[j] >= 0 || (bdy && bdy[j])) continue;

                    a[j] = agg;
                    frontier.push_back(j);
                }
            }
        }

        return {aggregates, n_aggregates};
    }

    template <typename R, typename C>
    SAVectorAMGAggregates<C> sa_vector_amg_optimal_aggregate(const ptrdiff_t        block_rows,
                                                             const SharedBuffer<R>& rowptr,
                                                             const SharedBuffer<C>& colidx,
                                                             const int              max_aggregate_size = 32) {
        auto     aggregates = create_host_buffer<C>(block_rows);
        C* const a          = aggregates->data();

        const R* const rp = rowptr->data();
        const C* const ci = colidx->data();

        std::vector<int> degree(block_rows, 0);
        std::vector<C>   order(block_rows);

        for (ptrdiff_t i = 0; i < block_rows; ++i) {
            a[i]     = static_cast<C>(-1);
            order[i] = static_cast<C>(i);

            for (R k = rp[i]; k < rp[i + 1]; ++k) {
                const C j = ci[k];
                degree[i] += (j >= 0 && static_cast<ptrdiff_t>(j) < block_rows && j != i);
            }
        }

        std::sort(order.begin(), order.end(), [&](const C left, const C right) {
            if (degree[left] != degree[right]) return degree[left] > degree[right];
            return left < right;
        });

        std::vector<C> members;
        std::vector<C> candidates;
        std::vector<R> candidate_mark(block_rows, static_cast<R>(-1));

        members.reserve(std::max(1, max_aggregate_size));
        candidates.reserve(std::max(1, max_aggregate_size * 4));

        ptrdiff_t n_aggregates = 0;

        for (const C seed : order) {
            if (a[seed] >= 0) continue;

            const C agg = static_cast<C>(n_aggregates++);
            a[seed]     = agg;

            members.clear();
            candidates.clear();
            members.push_back(seed);

            while (static_cast<int>(members.size()) < max_aggregate_size) {
                const R stamp = static_cast<R>(agg + 1);

                for (const C member : members) {
                    for (R k = rp[member]; k < rp[member + 1]; ++k) {
                        const C candidate = ci[k];
                        if (candidate < 0 || static_cast<ptrdiff_t>(candidate) >= block_rows || a[candidate] >= 0) continue;

                        if (candidate_mark[candidate] != stamp) {
                            candidate_mark[candidate] = stamp;
                            candidates.push_back(candidate);
                        }
                    }
                }

                C   best       = static_cast<C>(-1);
                int best_score = std::numeric_limits<int>::min();

                for (const C candidate : candidates) {
                    if (a[candidate] >= 0) continue;

                    int links_to_aggregate = 0;
                    for (R k = rp[candidate]; k < rp[candidate + 1]; ++k) {
                        const C neighbor = ci[k];
                        if (neighbor >= 0 && static_cast<ptrdiff_t>(neighbor) < block_rows && a[neighbor] == agg) {
                            ++links_to_aggregate;
                        }
                    }

                    const int score = links_to_aggregate * 1024 - degree[candidate];
                    if (score > best_score || (score == best_score && (best < 0 || candidate < best))) {
                        best       = candidate;
                        best_score = score;
                    }
                }

                if (best < 0) break;

                a[best] = agg;
                members.push_back(best);
            }
        }

        return {aggregates, n_aggregates};
    }

    template <typename R, typename C, typename T>
    std::shared_ptr<CRS<R, C, T, T>> sa_vector_amg_block_tentative_prolongation(const ptrdiff_t                 block_rows,
                                                                                const int                       block_size,
                                                                                const SAVectorAMGAggregates<C>& aggregates) {
        const ptrdiff_t fine_rows   = block_rows * block_size;
        const ptrdiff_t coarse_cols = aggregates.n_aggregates * block_size;

        auto rowptr = create_host_buffer<R>(fine_rows + 1);
        auto colidx = create_host_buffer<C>(fine_rows);
        auto values = create_host_buffer<T>(fine_rows);

        R* const       r   = rowptr->data();
        C* const       c   = colidx->data();
        T* const       v   = values->data();
        const C* const agg = aggregates.aggregate->data();

#pragma omp parallel for schedule(static)
        for (ptrdiff_t row = 0; row < fine_rows; ++row) {
            const ptrdiff_t node = row / block_size;
            const int       d    = row - node * block_size;

            r[row] = row;
            c[row] = agg[node] * block_size + d;
            v[row] = static_cast<T>(1);
        }

        r[fine_rows] = fine_rows;

        return h_crs_spmv<R, C, T, T>(fine_rows, coarse_cols, rowptr, colidx, values, static_cast<T>(0));
    }

    template <typename R, typename C, typename X, typename T>
    std::shared_ptr<CRS<R, C, T, T>> sa_vector_amg_tentative_prolongation(const ptrdiff_t                 block_rows,
                                                                          const int                       block_size,
                                                                          const int                       spatial_dim,
                                                                          const X* const* const           points,
                                                                          const SAVectorAMGAggregates<C>& aggregates) {
        const int       n_modes     = sa_vector_amg_n_rigid_body_modes(block_size, spatial_dim);
        const ptrdiff_t fine_rows   = block_rows * block_size;
        const ptrdiff_t coarse_cols = aggregates.n_aggregates * n_modes;

        auto rowptr = create_host_buffer<R>(fine_rows + 1);

        R* const       r   = rowptr->data();
        const C* const agg = aggregates.aggregate->data();

        r[0] = 0;
        for (ptrdiff_t node = 0; node < block_rows; ++node) {
            const R row_nnz = agg[node] >= 0 ? static_cast<R>(n_modes) : static_cast<R>(0);
            for (int d = 0; d < block_size; ++d) {
                r[node * block_size + d + 1] = row_nnz;
            }
        }

        for (ptrdiff_t i = 0; i < fine_rows; ++i) {
            r[i + 1] += r[i];
        }

        auto colidx = create_host_buffer<C>(r[fine_rows]);
        auto values = create_host_buffer<T>(r[fine_rows]);

        C* const c = colidx->data();
        T* const v = values->data();

        std::vector<T> centroid(aggregates.n_aggregates * spatial_dim, static_cast<T>(0));
        std::vector<T> count(aggregates.n_aggregates, static_cast<T>(0));

        for (ptrdiff_t node = 0; node < block_rows; ++node) {
            const C a = agg[node];
            if (a < 0) continue;
            count[a] += static_cast<T>(1);
            for (int d = 0; d < spatial_dim; ++d) {
                centroid[a * spatial_dim + d] += static_cast<T>(points[d][node]);
            }
        }

        for (ptrdiff_t a = 0; a < aggregates.n_aggregates; ++a) {
            const T inv_count = count[a] > 0 ? static_cast<T>(1) / count[a] : static_cast<T>(0);
            for (int d = 0; d < spatial_dim; ++d) {
                centroid[a * spatial_dim + d] *= inv_count;
            }
        }

        std::vector<T> gram(aggregates.n_aggregates * n_modes * n_modes, static_cast<T>(0));

        for (ptrdiff_t node = 0; node < block_rows; ++node) {
            const C a = agg[node];
            if (a < 0) continue;

            const T* const center = &centroid[a * spatial_dim];
            T* const       g      = &gram[a * n_modes * n_modes];

            for (int d = 0; d < block_size; ++d) {
                T q[6] = {0, 0, 0, 0, 0, 0};
                for (int m = 0; m < n_modes; ++m) {
                    q[m] = sa_vector_amg_rbm_value(block_size, spatial_dim, d, m, points, node, center);
                }

                for (int i = 0; i < n_modes; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        g[i * n_modes + j] += q[i] * q[j];
                    }
                }
            }
        }

        for (ptrdiff_t a = 0; a < aggregates.n_aggregates; ++a) {
            T* const g = &gram[a * n_modes * n_modes];
            for (int i = 0; i < n_modes; ++i) {
                for (int j = 0; j < i; ++j) {
                    g[j * n_modes + i] = g[i * n_modes + j];
                }
            }

            for (int k = 0; k < n_modes; ++k) {
                T diag = g[k * n_modes + k];
                for (int j = 0; j < k; ++j) {
                    diag -= g[k * n_modes + j] * g[k * n_modes + j];
                }

                g[k * n_modes + k] = diag > std::numeric_limits<T>::epsilon() ? std::sqrt(diag) : static_cast<T>(1);
                const T inv_diag   = static_cast<T>(1) / g[k * n_modes + k];

                for (int i = k + 1; i < n_modes; ++i) {
                    T val = g[i * n_modes + k];
                    for (int j = 0; j < k; ++j) {
                        val -= g[i * n_modes + j] * g[k * n_modes + j];
                    }
                    g[i * n_modes + k] = val * inv_diag;
                }
            }
        }

#pragma omp parallel for schedule(static)
        for (ptrdiff_t node = 0; node < block_rows; ++node) {
            const C a = agg[node];
            if (a < 0) continue;

            const T* const center = &centroid[a * spatial_dim];
            const T* const l      = &gram[a * n_modes * n_modes];

            for (int d = 0; d < block_size; ++d) {
                T q[6] = {0, 0, 0, 0, 0, 0};
                T y[6] = {0, 0, 0, 0, 0, 0};
                for (int m = 0; m < n_modes; ++m) {
                    q[m] = sa_vector_amg_rbm_value(block_size, spatial_dim, d, m, points, node, center);
                }

                for (int i = 0; i < n_modes; ++i) {
                    T val = q[i];
                    for (int j = 0; j < i; ++j) {
                        val -= l[i * n_modes + j] * y[j];
                    }
                    y[i] = val / l[i * n_modes + i];
                }

                const ptrdiff_t row = node * block_size + d;
                for (int m = 0; m < n_modes; ++m) {
                    const R offset = r[row] + m;
                    c[offset]      = a * n_modes + m;
                    v[offset]      = y[m];
                }
            }
        }

        return h_crs_spmv<R, C, T, T>(fine_rows, coarse_cols, rowptr, colidx, values, static_cast<T>(0));
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    std::shared_ptr<CRS<R, C, T, T>> sa_vector_amg_smooth_prolongation(const ptrdiff_t                         block_rows,
                                                                       const int                               block_size,
                                                                       const SharedBuffer<R>&                  bsr_rowptr,
                                                                       const SharedBuffer<C>&                  bsr_colidx,
                                                                       const SharedBuffer<TStorage>&           bsr_values,
                                                                       const std::shared_ptr<CRS<R, C, T, T>>& p,
                                                                       const T omega = static_cast<T>(4.0 / 3.0)) {
        const ptrdiff_t rows        = block_rows * block_size;
        const ptrdiff_t coarse_cols = p->cols();
        const int       block_area  = block_size * block_size;

        auto     rowptr = create_host_buffer<R>(rows + 1);
        R* const pr     = p->row_ptr->data();
        C* const pc     = p->col_idx->data();
        T* const pv     = p->values->data();

        const R* const        br = bsr_rowptr->data();
        const C* const        bc = bsr_colidx->data();
        const TStorage* const bv = bsr_values->data();

#ifdef _OPENMP
        const int n_workspaces = omp_get_max_threads();
#else
        const int n_workspaces = 1;
#endif
        std::vector<R> marker(n_workspaces * coarse_cols, static_cast<R>(-1));
        R* const       mr = marker.data();
        R* const       sr = rowptr->data();

        sr[0] = 0;

#pragma omp parallel num_threads(n_workspaces)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            R* const mark = &mr[tid * coarse_cols];

#pragma omp for schedule(static)
            for (ptrdiff_t row = 0; row < rows; ++row) {
                const ptrdiff_t node = row / block_size;
                R               nnz  = 0;

                for (R pk = pr[row]; pk < pr[row + 1]; ++pk) {
                    const C col = pc[pk];
                    if (mark[col] != row) {
                        mark[col] = row;
                        ++nnz;
                    }
                }

                for (R bk = br[node]; bk < br[node + 1]; ++bk) {
                    const C bj = bc[bk];
                    for (int d2 = 0; d2 < block_size; ++d2) {
                        const ptrdiff_t p_row = bj * block_size + d2;
                        for (R pk = pr[p_row]; pk < pr[p_row + 1]; ++pk) {
                            const C col = pc[pk];
                            if (mark[col] != row) {
                                mark[col] = row;
                                ++nnz;
                            }
                        }
                    }
                }

                sr[row + 1] = nnz;
            }
        }

        for (ptrdiff_t i = 0; i < rows; ++i) {
            sr[i + 1] += sr[i];
        }

        auto     colidx = create_host_buffer<C>(sr[rows]);
        auto     values = create_host_buffer<T>(sr[rows]);
        C* const sc     = colidx->data();
        T* const sv     = values->data();

        std::fill(marker.begin(), marker.end(), static_cast<R>(-1));

#pragma omp parallel num_threads(n_workspaces)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            R* const mark = &mr[tid * coarse_cols];

#pragma omp for schedule(static)
            for (ptrdiff_t row = 0; row < rows; ++row) {
                const ptrdiff_t node = row / block_size;
                const int       d1   = row - node * block_size;
                R               end  = sr[row];
                T               diag = static_cast<T>(0);

                for (R bk = br[node]; bk < br[node + 1]; ++bk) {
                    if (bc[bk] == node) {
                        diag = static_cast<T>(bv[bk * block_area + d1 * block_size + d1]);
                        break;
                    }
                }

                if (std::abs(diag) <= std::numeric_limits<T>::epsilon()) {
                    diag = static_cast<T>(1);
                }

                for (R pk = pr[row]; pk < pr[row + 1]; ++pk) {
                    const C col = pc[pk];
                    mark[col]   = end;
                    sc[end]     = col;
                    sv[end]     = pv[pk];
                    ++end;
                }

                const T scale = -omega / diag;
                for (R bk = br[node]; bk < br[node + 1]; ++bk) {
                    const C bj = bc[bk];
                    for (int d2 = 0; d2 < block_size; ++d2) {
                        const T aij = static_cast<T>(bv[bk * block_area + d1 * block_size + d2]);
                        if (aij == static_cast<T>(0)) continue;

                        const ptrdiff_t p_row = bj * block_size + d2;
                        const T         coeff = scale * aij;
                        for (R pk = pr[p_row]; pk < pr[p_row + 1]; ++pk) {
                            const C col = pc[pk];
                            R       pos = mark[col];
                            if (pos == static_cast<R>(-1)) {
                                pos       = end;
                                mark[col] = pos;
                                sc[end]   = col;
                                sv[end]   = coeff * pv[pk];
                                ++end;
                            } else {
                                sv[pos] += coeff * pv[pk];
                            }
                        }
                    }
                }

                for (R k = sr[row]; k < end; ++k) {
                    mark[sc[k]] = static_cast<R>(-1);
                }
            }
        }

        return h_crs_spmv<R, C, T, T>(rows, coarse_cols, rowptr, colidx, values, static_cast<T>(0));
    }

    template <typename R, typename C, typename T>
    std::shared_ptr<CRS<R, C, T, T>> sa_vector_amg_restriction(const std::shared_ptr<CRS<R, C, T, T>>& p) {
        return p->transpose();
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    std::shared_ptr<CRS<R, C, T, T>> sa_vector_amg_bsr_matmul(const ptrdiff_t                         block_rows,
                                                              const int                               block_size,
                                                              const SharedBuffer<R>&                  bsr_rowptr,
                                                              const SharedBuffer<C>&                  bsr_colidx,
                                                              const SharedBuffer<TStorage>&           bsr_values,
                                                              const std::shared_ptr<CRS<R, C, T, T>>& x) {
        const ptrdiff_t rows       = block_rows * block_size;
        const int       block_area = block_size * block_size;
        const ptrdiff_t columns    = x->cols();

        auto rowptr = create_host_buffer<R>(rows + 1);

        const R* const        br = bsr_rowptr->data();
        const C* const        bc = bsr_colidx->data();
        const TStorage* const bv = bsr_values->data();

        const R* const xr = x->row_ptr->data();
        const C* const xc = x->col_idx->data();
        const T* const xv = x->values->data();
        R* const       yr = rowptr->data();

#ifdef _OPENMP
        const int n_workspaces = omp_get_max_threads();
#else
        const int n_workspaces = 1;
#endif

        std::vector<R> marker(n_workspaces * columns, static_cast<R>(-1));
        R* const       mr = marker.data();

        yr[0] = 0;

#pragma omp parallel num_threads(n_workspaces)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            R* const mark = &mr[tid * columns];

#pragma omp for schedule(static)
            for (ptrdiff_t row = 0; row < rows; ++row) {
                const ptrdiff_t node = row / block_size;
                R               nnz  = 0;

                for (R bk = br[node]; bk < br[node + 1]; ++bk) {
                    const C bj = bc[bk];
                    for (int d2 = 0; d2 < block_size; ++d2) {
                        const ptrdiff_t x_row = bj * block_size + d2;
                        for (R xk = xr[x_row]; xk < xr[x_row + 1]; ++xk) {
                            const C col = xc[xk];
                            if (mark[col] != row) {
                                mark[col] = row;
                                ++nnz;
                            }
                        }
                    }
                }

                yr[row + 1] = nnz;
            }
        }

        for (ptrdiff_t i = 0; i < rows; ++i) {
            yr[i + 1] += yr[i];
        }

        auto     colidx = create_host_buffer<C>(yr[rows]);
        auto     values = create_host_buffer<T>(yr[rows]);
        C* const yc     = colidx->data();
        T* const yv     = values->data();

        std::fill(marker.begin(), marker.end(), static_cast<R>(-1));

#pragma omp parallel num_threads(n_workspaces)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            R* const mark = &mr[tid * columns];

#pragma omp for schedule(static)
            for (ptrdiff_t row = 0; row < rows; ++row) {
                const ptrdiff_t node = row / block_size;
                const int       d1   = row - node * block_size;
                R               end  = yr[row];

                for (R bk = br[node]; bk < br[node + 1]; ++bk) {
                    const C bj = bc[bk];
                    for (int d2 = 0; d2 < block_size; ++d2) {
                        const T aij = static_cast<T>(bv[bk * block_area + d1 * block_size + d2]);
                        if (aij == static_cast<T>(0)) continue;

                        const ptrdiff_t x_row = bj * block_size + d2;
                        for (R xk = xr[x_row]; xk < xr[x_row + 1]; ++xk) {
                            const C col = xc[xk];
                            R       pos = mark[col];
                            if (pos == static_cast<R>(-1)) {
                                pos       = end;
                                mark[col] = pos;
                                yc[end]   = col;
                                yv[end]   = aij * xv[xk];
                                ++end;
                            } else {
                                yv[pos] += aij * xv[xk];
                            }
                        }
                    }
                }

                for (R k = yr[row]; k < end; ++k) {
                    mark[yc[k]] = static_cast<R>(-1);
                }
            }
        }

        return h_crs_spmv<R, C, T, T>(rows, columns, rowptr, colidx, values, static_cast<T>(0));
    }

    template <typename R, typename C, typename T>
    std::shared_ptr<BSR<R, C, T, T>> sa_vector_amg_galerkin_bsr(const std::shared_ptr<CRS<R, C, T, T>>& r,
                                                                const std::shared_ptr<CRS<R, C, T, T>>& ap,
                                                                const int                               coarse_block_size) {
        const ptrdiff_t block_rows = r->rows() / coarse_block_size;
        const ptrdiff_t block_cols = ap->cols() / coarse_block_size;
        const int       block_area = coarse_block_size * coarse_block_size;

        auto rowptr = create_host_buffer<R>(block_rows + 1);

        const R* const rr = r->row_ptr->data();
        const C* const rc = r->col_idx->data();
        const T* const rv = r->values->data();

        const R* const ar = ap->row_ptr->data();
        const C* const ac = ap->col_idx->data();
        const T* const av = ap->values->data();

        R* const br = rowptr->data();

#ifdef _OPENMP
        const int n_workspaces = omp_get_max_threads();
#else
        const int n_workspaces = 1;
#endif

        std::vector<R> marker(n_workspaces * block_cols, static_cast<R>(-1));
        R* const       mr = marker.data();

        br[0] = 0;

#pragma omp parallel num_threads(n_workspaces)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            R* const mark = &mr[tid * block_cols];

#pragma omp for schedule(static)
            for (ptrdiff_t bi = 0; bi < block_rows; ++bi) {
                const R stamp = static_cast<R>(bi);
                R       nnz   = 0;

                for (int m = 0; m < coarse_block_size; ++m) {
                    const ptrdiff_t r_row = bi * coarse_block_size + m;
                    for (R rk = rr[r_row]; rk < rr[r_row + 1]; ++rk) {
                        const C fine_row = rc[rk];
                        for (R ak = ar[fine_row]; ak < ar[fine_row + 1]; ++ak) {
                            const C bj = ac[ak] / coarse_block_size;
                            if (mark[bj] != stamp) {
                                mark[bj] = stamp;
                                ++nnz;
                            }
                        }
                    }
                }

                br[bi + 1] = nnz;
            }
        }

        for (ptrdiff_t bi = 0; bi < block_rows; ++bi) {
            br[bi + 1] += br[bi];
        }

        auto colidx = create_host_buffer<C>(br[block_rows]);
        auto values = create_host_buffer<T>(br[block_rows] * block_area);

        C* const bc = colidx->data();
        T* const bv = values->data();

        std::fill(marker.begin(), marker.end(), static_cast<R>(-1));

#pragma omp parallel num_threads(n_workspaces)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            R* const mark = &mr[tid * block_cols];

#pragma omp for schedule(static)
            for (ptrdiff_t bi = 0; bi < block_rows; ++bi) {
                R end = br[bi];

                for (int m = 0; m < coarse_block_size; ++m) {
                    const ptrdiff_t r_row = bi * coarse_block_size + m;
                    for (R rk = rr[r_row]; rk < rr[r_row + 1]; ++rk) {
                        const C fine_row = rc[rk];
                        const T r_val    = rv[rk];

                        for (R ak = ar[fine_row]; ak < ar[fine_row + 1]; ++ak) {
                            const C   scalar_col = ac[ak];
                            const C   bj         = scalar_col / coarse_block_size;
                            const int n          = scalar_col - bj * coarse_block_size;

                            R pos = mark[bj];
                            if (pos == static_cast<R>(-1)) {
                                pos            = end;
                                mark[bj]       = pos;
                                bc[end]        = bj;
                                T* const block = &bv[end * block_area];
                                for (int k = 0; k < block_area; ++k) {
                                    block[k] = static_cast<T>(0);
                                }
                                ++end;
                            }

                            bv[pos * block_area + m * coarse_block_size + n] += r_val * av[ak];
                        }
                    }
                }

                for (R k = br[bi]; k < end; ++k) {
                    mark[bc[k]] = static_cast<R>(-1);
                }
            }
        }

        return h_bsr_spmv<R, C, T, T>(block_rows, block_cols, coarse_block_size, rowptr, colidx, values, static_cast<T>(0));
    }

    template <typename R, typename C, typename TStorage, typename T = TStorage>
    std::shared_ptr<BSR<R, C, T, T>> sa_vector_amg_coarse_matrix(const ptrdiff_t                         block_rows,
                                                                 const int                               block_size,
                                                                 const int                               coarse_block_size,
                                                                 const SharedBuffer<R>&                  bsr_rowptr,
                                                                 const SharedBuffer<C>&                  bsr_colidx,
                                                                 const SharedBuffer<TStorage>&           bsr_values,
                                                                 const std::shared_ptr<CRS<R, C, T, T>>& r,
                                                                 const std::shared_ptr<CRS<R, C, T, T>>& p) {
        auto ap = sa_vector_amg_bsr_matmul<R, C, TStorage, T>(block_rows, block_size, bsr_rowptr, bsr_colidx, bsr_values, p);
        return sa_vector_amg_galerkin_bsr<R, C, T>(r, ap, coarse_block_size);
    }

    template <typename R, typename C, typename TStorage, typename X, typename T = TStorage>
    SAVectorAMGLevel<R, C, TStorage, T> h_sa_vector_amg_level(const std::shared_ptr<BSR<R, C, TStorage, T>>& a_bsr,
                                                              const X* const* const                          points,
                                                              const int                                      spatial_dim,
                                                              const SharedBuffer<mask_t>& boundary_nodes     = nullptr,
                                                              const int                   max_aggregate_size = 8,
                                                              const T prolongation_omega = static_cast<T>(4.0 / 3.0)) {
        SAVectorAMGLevel<R, C, TStorage, T> level;

        const ptrdiff_t block_rows = a_bsr->row_ptr->size() - 1;
        const int       block_size = a_bsr->block_size();

        level.aggregates =
                sa_vector_amg_aggregate<R, C>(block_rows, a_bsr->row_ptr, a_bsr->col_idx, boundary_nodes, max_aggregate_size);
        level.n_rigid_body_modes = sa_vector_amg_n_rigid_body_modes(block_size, spatial_dim);
        auto tentative_p =
                sa_vector_amg_tentative_prolongation<R, C, X, T>(block_rows, block_size, spatial_dim, points, level.aggregates);
        level.p = sa_vector_amg_smooth_prolongation<R, C, TStorage, T>(
                block_rows, block_size, a_bsr->row_ptr, a_bsr->col_idx, a_bsr->values, tentative_p, prolongation_omega);
        level.r        = sa_vector_amg_restriction<R, C, T>(level.p);
        level.a        = a_bsr;
        level.coarse_a = sa_vector_amg_coarse_matrix<R, C, TStorage, T>(block_rows,
                                                                        block_size,
                                                                        level.n_rigid_body_modes,
                                                                        a_bsr->row_ptr,
                                                                        a_bsr->col_idx,
                                                                        a_bsr->values,
                                                                        level.r,
                                                                        level.p);

        return level;
    }

    template <typename R, typename C, typename T>
    SAVectorAMGLevel<R, C, T, T> h_sa_vector_amg_coarse_level(const std::shared_ptr<BSR<R, C, T, T>>& a_bsr,
                                                              const int                               max_aggregate_size = 32,
                                                              const T prolongation_omega = static_cast<T>(4.0 / 3.0)) {
        SAVectorAMGLevel<R, C, T, T> level;

        const ptrdiff_t block_rows = a_bsr->row_ptr->size() - 1;
        const int       block_size = a_bsr->block_size();

        level.aggregates = sa_vector_amg_optimal_aggregate<R, C>(block_rows, a_bsr->row_ptr, a_bsr->col_idx, max_aggregate_size);
        level.n_rigid_body_modes = block_size;

        auto tentative_p = sa_vector_amg_block_tentative_prolongation<R, C, T>(block_rows, block_size, level.aggregates);
        level.p          = sa_vector_amg_smooth_prolongation<R, C, T, T>(
                block_rows, block_size, a_bsr->row_ptr, a_bsr->col_idx, a_bsr->values, tentative_p, prolongation_omega);
        level.r        = sa_vector_amg_restriction<R, C, T>(level.p);
        level.a        = a_bsr;
        level.coarse_a = sa_vector_amg_coarse_matrix<R, C, T, T>(
                block_rows, block_size, block_size, a_bsr->row_ptr, a_bsr->col_idx, a_bsr->values, level.r, level.p);

        return level;
    }

    template <typename R, typename C, typename X, typename T>
    SAVectorAMGHierarchy<R, C, T> h_sa_vector_amg_hierarchy(const std::shared_ptr<BSR<R, C, T, T>>& a_bsr,
                                                            const X* const* const                   points,
                                                            const int                               spatial_dim,
                                                            const SharedBuffer<mask_t>&             boundary_nodes,
                                                            const int                               fine_max_aggregate_size = 32,
                                                            const int       coarse_max_aggregate_size                       = 32,
                                                            const int       max_levels                                      = 4,
                                                            const ptrdiff_t coarsest_block_rows                             = 32,
                                                            const T         prolongation_omega = static_cast<T>(4.0 / 3.0)) {
        SAVectorAMGHierarchy<R, C, T> hierarchy;

        if (max_levels <= 0) return hierarchy;

        auto fine_level = h_sa_vector_amg_level<R, C, T, X, T>(
                a_bsr, points, spatial_dim, boundary_nodes, fine_max_aggregate_size, prolongation_omega);

        hierarchy.levels.push_back(fine_level);

        auto current = fine_level.coarse_a;

        while (static_cast<int>(hierarchy.levels.size()) < max_levels && current->row_ptr->size() > 1) {
            const ptrdiff_t current_block_rows = current->row_ptr->size() - 1;
            if (current_block_rows <= coarsest_block_rows) break;

            auto coarse_level = h_sa_vector_amg_coarse_level<R, C, T>(current, coarse_max_aggregate_size, prolongation_omega);

            if (coarse_level.aggregates.n_aggregates >= current_block_rows) break;

            hierarchy.levels.push_back(coarse_level);
            current = coarse_level.coarse_a;
        }

        return hierarchy;
    }
}  // namespace sfem

#endif  // SFEM_SA_VECTOR_AMG_HPP
