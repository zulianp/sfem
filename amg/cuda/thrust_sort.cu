#include "sfem_base.h"
#ifdef SFEM_ENABLE_CUDA

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include "coo_sort.h"

struct CompareTuplesWeights {
    __host__ __device__ bool operator()(const thrust::tuple<idx_t, idx_t, real_t> &a,
                                        const thrust::tuple<idx_t, idx_t, real_t> &b) const {
        return thrust::get<2>(a) > thrust::get<2>(b);  // Descending order
    }
};

struct CompareTuplesIndices {
    __host__ __device__ bool operator()(const thrust::tuple<idx_t, idx_t, real_t> &a,
                                        const thrust::tuple<idx_t, idx_t, real_t> &b) const {
        idx_t a_row = thrust::get<0>(a);
        idx_t b_row = thrust::get<0>(b);

        if (a_row != b_row) {
            return a_row < b_row;
        } else {
            idx_t a_col = thrust::get<1>(a);
            idx_t b_col = thrust::get<1>(b);
            return a_col < b_col;
        }
    }
};

void sort_weights(count_t *sort_indices, idx_t *row_indices, idx_t *col_indices, real_t *weights,
                  const count_t N) {
    thrust::device_vector<idx_t> d_col_indices(col_indices, col_indices + N);
    thrust::device_vector<idx_t> d_row_indices(row_indices, row_indices + N);
    thrust::device_vector<real_t> d_values(weights, weights + N);

    auto first = thrust::make_zip_iterator(
            thrust::make_tuple(d_row_indices.begin(), d_col_indices.begin(), d_values.begin()));
    auto last = thrust::make_zip_iterator(
            thrust::make_tuple(d_row_indices.end(), d_col_indices.end(), d_values.end()));

    thrust::stable_sort(first, last, CompareTuplesWeights());

    thrust::copy(d_row_indices.begin(), d_row_indices.end(), row_indices);
    thrust::copy(d_col_indices.begin(), d_col_indices.end(), col_indices);
    thrust::copy(d_values.begin(), d_values.end(), weights);
}

void sort_rows_cols(count_t *sort_indices, idx_t *row_indices, idx_t *col_indices, real_t *weights,
                    const count_t N) {
    thrust::device_vector<idx_t> d_col_indices(col_indices, col_indices + N);
    thrust::device_vector<idx_t> d_row_indices(row_indices, row_indices + N);
    thrust::device_vector<real_t> d_values(weights, weights + N);

    auto first = thrust::make_zip_iterator(
            thrust::make_tuple(d_row_indices.begin(), d_col_indices.begin(), d_values.begin()));
    auto last = thrust::make_zip_iterator(
            thrust::make_tuple(d_row_indices.end(), d_col_indices.end(), d_values.end()));

    thrust::stable_sort(first, last, CompareTuplesIndices());

    thrust::copy(d_row_indices.begin(), d_row_indices.end(), row_indices);
    thrust::copy(d_col_indices.begin(), d_col_indices.end(), col_indices);
    thrust::copy(d_values.begin(), d_values.end(), weights);
}
#endif
